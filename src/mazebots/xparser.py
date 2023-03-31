"""Simple Xacro URDF parser for specific use"""

import re
import sys


class XacroParser:
    def __init__(self):
        self.vars: 'dict[str, float]' = {}

    def parse(self, path_in: str, path_out: str):
        """Parse a file from Xacro into raw URDF, line by line."""

        with open(path_in, 'r') as file_in:
            with open(path_out, 'w') as file_out:
                for line in file_in:
                    line = self.parse_line(line.strip())

                    if line:
                        file_out.write(line)

    def parse_line(self, line: str) -> str:
        """Parse a line from Xacro into raw URDF."""

        if not line or any(line.startswith(x) for x in ('<!--', '-->', '<xacro:macro', '</xacro:macro', '<inertia ')):
            return ''

        if line.startswith('<xacro:property'):
            name = re.search(r'name="(.*?)"', line).group(1)
            val = re.search(r'value="(.*?)"', line).group(1)

            self.vars[name] = self.eval_exp(val)

            return ''

        if line.startswith('<xacro:box'):
            args = [self.eval_exp(re.search(rf'{arg}="(.*?)"', line).group(1)) for arg in ('mass', 'x', 'y', 'z')]

            line = '<inertia ixx="%f" ixy="%f" ixz="%f" iyy="%f" iyz="%f" izz="%f"/>\n' % box_inertia(*args)

        elif line.startswith('<xacro:cyl'):
            args = [self.eval_exp(re.search(rf'{arg}="(.*?)"', line).group(1)) for arg in ('mass', 'rad', 'len')]

            line = '<inertia ixx="%f" ixy="%f" ixz="%f" iyy="%f" iyz="%f" izz="%f"/>\n' % cyl_inertia(*args)

        else:
            line = self.replace_exps(line) + '\n'

        return line

    def replace_exps(self, line: str) -> str:
        """Replace expressions in a line with their evaluations."""

        new_line = ''
        exp = ''
        exp_switch = False

        for char in line:
            if char == '$':
                exp = char
                exp_switch = True

            elif char == '}':
                new_line += str(self.eval_exp(exp + char))
                exp = ''
                exp_switch = False

            elif exp_switch:
                exp += char

            else:
                new_line += char

        return new_line

    def eval_exp(self, exp: str) -> float:
        """Evaluate an expression with abstract terms."""

        if exp.startswith('$'):
            exp = exp[2:-1]

        for key, val in self.vars.items():
            if key in exp:
                exp = exp.replace(key, str(val))

        return round(eval(exp), 10)


def box_inertia(m: float, x: float, y: float, z: float) -> 'tuple[float, ...]':
    """Calculate box inertia."""

    return (
        (y**2 + z**2) * m/12,
        0.,
        0.,
        (x**2 + z**2) * m/12,
        0.,
        (x**2 + y**2) * m/12)


def cyl_inertia(m: float, r: float, h: float) -> 'tuple[float, ...]':
    """Calculate cylinder inertia."""

    return (
        (3*r**2 + h**2) * m/12,
        0.,
        0.,
        (3*r**2 + h**2) * m/12,
        0.,
        r**2 * m/2)


if __name__ == '__main__' and len(sys.argv) > 1:
    path_in = sys.argv[1]
    path_out = sys.argv[2] if len(sys.argv) > 2 else f'{path_in[:-5]}_parsed.urdf'

    XacroParser().parse(path_in, path_out)
