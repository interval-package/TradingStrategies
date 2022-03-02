import scipy.optimize as sp
import numpy as np


def funcGetNextDay(Sws, ms, gs, Cs=None):
    # gs[2] should == 0
    gs.append(0)
    if Cs is None:
        Cs = [0.02, 0.01, 0]

    result = []

    # 卖出预付佣金
    # 买入的佣金从卖出中扣去
    for i in range(3):
        result.append((ms[i] - Sws[2 * i] * (1 + Cs[i]) + Sws[2 * i + 1] * (1 - Cs[i])) * (1 + gs[i]))
    return result


def coefGet(g, c, caseFlag=True):
    if caseFlag:
        return -(1 + g) * (1 + c)
    else:
        return (1 + g) * (1 - c)


def getNextDayCoef(ms, gs, Cs=None, bound=None):
    if Cs is None:
        Cs = [0.02, 0.01, 0]
    if bound is None:
        bound = [0.5, 0.2, 0.8]
    ms = np.array(ms)
    gs = np.array(gs)
    Cs = np.array(Cs)

    A = []
    for i in range(len(Cs)):
        A.append(coefGet(gs[i], Cs[i], True))
        A.append(coefGet(gs[i], Cs[i], False))
    A = np.array(A)
    B = np.ones((1, 6))
    for i in range(len(B[0])):
        if i % 2 == 0:
            B[0][i] = 1+Cs[int(i/2)]
        else:
            B[0][i] = -1
    Beq = np.array([0])
    C_up_eq = (np.array([bound]) * np.array(ms)).T
    C_lu = []
    for i in range(3):
        C_lu.append((0, 0.8*ms[i]))
        C_lu.append((0, C_up_eq[i][0]))
    return A, B, Beq, C_lu


def optimMyTarget(ms, gs, bound=None, Cs=None, ):
    A, B, Beq, C_lu = getNextDayCoef(ms, gs, Cs, bound)
    A = -A
    res = sp.linprog(c=A, A_eq=B, b_eq=Beq, bounds=C_lu)
    x = res.x
    return x


if __name__ == '__main__':
    # print(getNextDayCoef([123, 123, 123], [0.2, 0.05, 0]))
    cm = [0.02, 0.01, 0]
    trans = optimMyTarget([100, 100, 800], [0.4, 0.05, 0], [100, 100, 10])
    print(trans)
    print(-trans[0]*(1+cm[0])+trans[1])

    pass
