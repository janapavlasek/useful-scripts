import sys
import re


class Link(object):
    def __init__(self, name, xyz=[0, 0, 0], rpy=[0, 0, 0], filename=""):
        self.name = name
        self.filename = filename
        self.xyz = xyz
        self.rpy = rpy

    def __str__(self):
        return '\n'.join(["\tname: {}".format(self.name),
                          "\tfilename: {}".format(self.filename),
                          "\torigin: xyz = {}, rpy = {}".format(self.xyz, self.rpy)])


class Joint(object):
    def __init__(self, parent, child, type, xyz, rpy, axis=None, limits=None):
        self.parent = parent
        self.child = child
        self.type = type
        self.xyz = xyz
        self.rpy = rpy
        self.axis = axis
        self.limits = limits
        self.value = 0

    def __str__(self):
        data = ["\tparent: {}".format(self.parent),
                "\tchild: {}".format(self.child),
                "\ttype: {}".format(self.type),
                "\torigin: xyz = {}, rpy = {}".format(self.xyz, self.rpy)]

        if self.axis is not None:
            data.append("\taxis: {}".format(self.axis))
        if self.limits is not None:
            data.append("\tlimits: {}".format(self.limits))

        return '\n'.join(data)


class URDF(object):
    def __init__(self, name, root=None):
        self.name = name
        self.root = root
        self.links = {}
        self.joints = []
        self.num_active_joints = 0

    def __str__(self):
        s = "name: {}\nroot: {}\n".format(self.name, self.root)
        s += "links:\n"
        links = []
        for k, v in self.links.items():
            links.append("{}:\n{}".format(k, v))
        s += "\n".join(links)
        s += "\njoints:\n" + '\n\t---------------\n'.join(str(j) for j in self.joints)
        return s

    def link_names(self):
        return self.links.keys()

    def add_link(self, name, link):
        self.links.update({name: link})

        if self.root is None:
            self.root = name

    def add_joint(self, joint):
        self.joints.append(joint)
        if joint.type != "fixed":
            self.num_active_joints += 1

    def fk(self):
        pass


def get_header_data(tag, s):
    """Gets the header data from a tag. For example, for tag robot:

        <robot name="robot_name">
            ...
        </robot>

    Function will return string: name="robot_name"
    """
    data = re.search('(?<=<{tag})([^<]*)(?=>)'.format(tag=tag), s)
    if data:
        return data.group(0).strip()


def get_tag_data(tag, s):
    """Gets the data from a tag. It will first assume a multiline tag, between
    markers <tag>...</tag>, and if that fails, try a single line tag, between
    <tag .../>. Doesn't return the tags themselves. If the tag appears multiple
    times in the string, the first one is returned."""
    # Try multiline tag first.
    data = re.search('(?<=<{tag})(.*?)(?=<\\/{tag}>)'.format(tag=tag), s)
    if data:
        return data.group(0)

    # Try single line tag next.
    data = re.search('(?<=<{tag})([^<]*)(?=\\/>)'.format(tag=tag), s)
    if data:
        return data.group(0)


def get_origin(s):
    """Gets the xyz and rpy from an origin tag, as two lists of floats."""
    # Grab origin part.
    origin = get_tag_data("origin", s)
    # Find position and rotation strings from origin part.
    rpy = re.search('(?<=rpy=")(.*?)(?=")', origin)
    rpy = rpy.group(0).strip().split(' ') if rpy else [0, 0, 0]
    xyz = re.search('(?<=xyz=")(.*?)(?=")', origin)
    xyz = xyz.group(0).strip().split(' ') if xyz else [0, 0, 0]
    # Convert to floats.
    rpy = [float(ele) for ele in rpy]
    xyz = [float(ele) for ele in xyz]

    return xyz, rpy


def parse_urdf(urdf_path):
    with open(urdf_path, 'r') as f:
        lines = ''.join(f.readlines())
        lines = lines.replace('\n', '')

    name = re.search('(?<=name=")(.*?)(?=")', get_header_data("robot", lines)).group(0)
    obj_data = URDF(name)

    # Check for links without a body.
    empty_links = re.findall('<link[^<]*\\/>', lines)
    for m in empty_links:
        name = re.search('(?<=name=")(.*?)(?=")', m).group(0)
        obj_data.add_link(name, Link(name))
        lines = lines.replace(m, "")

    # Find all the links.
    link_matches = re.findall('<link.*?<\\/link>', lines)  # (?<=<link)(.*?)(?=<\\/link>)
    for m in link_matches:
        name = re.search('(?<=name=")(.*?)(?=")', get_header_data("link", m)).group(0)

        m = get_tag_data("visual", m)
        # Get filename.
        filename = re.search('(?<=filename=")(.*?)(?=")', m).group(0).replace("file://", "")

        # Get the origin element.
        xyz, rpy = get_origin(m)

        link = Link(name, xyz, rpy, filename)
        obj_data.add_link(name, link)

    joint_matches = re.findall('(?<=<joint)(.*?)(?=<\\/joint>)', lines)
    for m in joint_matches:
        type = re.search('(?<=type=")(.*?)(?=")', m).group(0).lower()
        # Get parent and child.
        parent = re.search('(?<=link=")(.*?)(?=")', get_tag_data("parent", m)).group(0)
        child = re.search('(?<=link=")(.*?)(?=")', get_tag_data("child", m)).group(0)

        # Get the origin element.
        xyz, rpy = get_origin(m)

        joint = Joint(parent, child, type, xyz, rpy)

        # There has to be an axis element if the joint isn't fixed.
        if type != "fixed":
            # The axis information.
            axis = get_tag_data("axis", m)
            axis = re.search('(?<=xyz=")(.*?)(?=")', axis).group(0).strip().split(' ')
            axis = [int(ele) for ele in axis]

            joint.axis = axis

            # If the joint isn't fixed, there might be limits.
            lim = get_tag_data("limit", m)
            if lim:
                lower = re.search('(?<=lower=")(.*?)(?=")', lim).group(0)
                upper = re.search('(?<=upper=")(.*?)(?=")', lim).group(0)
                joint.limits = [float(lower), float(upper)]

        obj_data.add_joint(joint)

    return obj_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        URDF_PATH = sys.argv[1]
    else:
        URDF_PATH = "/home/jana/progress/data/mesh_models/clamp/clamp.urdf"

    urdf = parse_urdf(URDF_PATH)
    print(urdf)
