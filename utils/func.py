from functools import partial


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def prog4(out1, out2, out3, out4):
    return partial(progn, out1, out2, out3, out4)


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()