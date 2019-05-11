class Params(object):
    def __init__(self) -> None:
        self.dataset = None
        self.model = None

        self.degree = 0
        self.compositions_num = 0

        self.mu = None
        self.threeL = False


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_config(configfile: str) -> Params:
    params = Params()
    with open(configfile, "r") as f:
        for l in f.readlines():
            line = l[:-1]
            varname, value = line.split("=")
            if is_number(value):
                if "." in value:
                    cmd = "params.%s = %f" % (varname, float(value))
                else:
                    cmd = "params.%s = %d" % (varname, int(value))
            elif value in ["True", "False"]:
                cmd = "params.%s = %s" % (varname, value)
            else:
                cmd = "params.%s = '%s'" % (varname, value)
            exec(cmd)
    return params