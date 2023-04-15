import numpy as np
import matplotlib.pyplot as plt
import math


def func(x):
    return np.sin(x) * (x ** 2)


def dichotomy_minimization(func, a, b, eps):
    num_evals = 0
    num_iters = 0
    interval_lengths = [b - a]

    while abs(b - a) / 2 > eps:
        num_iters += 1
        x1 = (a + b - eps) / 2
        x2 = (a + b + eps) / 2
        f1, f2 = func(x1), func(x2)
        num_evals += 2
        if f1 < f2:
            b = x2
        else:
            a = x1
        interval_lengths.append(b - a)

    return (a + b) / 2, num_iters, num_evals, interval_lengths


def golden_ratio_minimization(func, a, b, tol):
    golden_ratio = (1 + math.sqrt(5)) / 2

    x1 = b - (b - a) / golden_ratio
    x2 = a + (b - a) / golden_ratio

    f1, f2 = func(x1), func(x2)

    num_evals = 2
    num_iters = 0
    interval_lengths = [b - a]

    while abs(b - a) / 2 > tol:
        num_iters += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / golden_ratio
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) / golden_ratio
            f2 = func(x2)

        num_evals += 1
        interval_lengths.append(b - a)

    return (a + b) / 2, num_iters, num_evals, interval_lengths


def fibonacci_minimization(f, a, b, eps):
    # Определим последовательность Фибоначчи
    fib_seq = [1, 1]
    while fib_seq[-1] < (b - a) / eps:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])

    n = len(fib_seq) - 2
    x1 = a + (b - a) * fib_seq[n - 1] / fib_seq[n + 1]
    x2 = a + (b - a) * fib_seq[n] / fib_seq[n + 1]
    fx1 = f(x1)
    fx2 = f(x2)

    num_iter = 0
    num_evals = 2
    interval_lengths = [b - a]

    # Итерируем до сходимости
    while n > 1:
        num_iter += 1
        n -= 1
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + (b - a) * fib_seq[n - 1] / fib_seq[n + 1]
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + (b - a) * fib_seq[n] / fib_seq[n + 1]
            fx2 = f(x2)
        num_evals += 1
        interval_lengths.append(b - a)

    return (a + b) / 2, num_iter, num_evals, interval_lengths


def parabolic_minimization(func, a, b, tol):
    x = (a + b) / 2
    fx = func(x)
    l = a
    fl = func(l)
    r = b
    fr = func(r)

    interval_lengths = [b - a]
    num_iters = 0
    num_evals = 3

    while r - l > tol:
        denominator = (2 * ((x - l) * (fx - fr) - (x - r) * (fx - fl)))
        if denominator == 0:
            u = (r + l) / 2
        else:
            u = x - ((x - l) ** 2 * (fx - fr) - (x - r) ** 2 * (fx - fl)) / denominator
        fu = func(u)

        if fu > fx:
            if u > x:
                r = u
                fr = fu
            else:
                l = u
                fl = fu
        else:
            if x > u:
                r = x
                fr = fx
            else:
                l = x
                fl = fx
            x = u
            fx = fu

        interval_lengths.append(r - l)
        num_iters += 1
        num_evals += 1

    return (l + r) / 2, num_iters, num_evals, interval_lengths


def brent_minimization(func, left, right, tol):
    eps = 1e-14  # малая константа для избежания деления на ноль
    rho = (3 - math.sqrt(5)) / 2
    a = b = x = w = v = left
    d = e = 0
    fx = fw = fv = func(left)

    num_iters = num_evals = 0
    intervals_lengths = []

    while True:
        num_iters += 1
        xm = 0.5 * (left + right)
        tol1 = tol * abs(x) + eps
        tol2 = 2 * tol1
        if abs(x - xm) <= (tol2 - 0.5 * (right - left)):
            break
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (left - x) or p >= q * (right - x):
                if x >= xm:
                    e = left - x
                else:
                    e = right - x
                d = rho * e
            else:
                d = p / q
                u = x + d
                if u - left < tol2 or right - u < tol2:
                    d = tol1 * math.copysign(1, xm - x)
        else:
            if x >= xm:
                e = left - x
            else:
                e = right - x
            d = rho * e
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + tol1 * math.copysign(1, d)
        fu = func(u)
        num_evals += 1
        if fu <= fx:
            if u >= x:
                left = x
            else:
                right = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u >= x:
                right = u
            else:
                left = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

        intervals_lengths.append(right - left)

    return x, num_iters, num_evals, intervals_lengths


def print_stats(func, a, b, tol, minimization):
    result, num_iter, num_evals, interval_lengths = minimization(func, a, b, tol)
    print(f"Полученный минимум = {result:0.4f}")
    print(f"Количество итераций = {num_iter}")
    print(f"Количество вычисления функции = {num_evals}")
    plt.xlabel("Номер итерации")
    plt.ylabel("Длина отрезка")
    plt.plot(interval_lengths, marker='.')
    plt.show()


def unimodal():
    a = -4
    b = 2
    tol = 0.001
    x = [x for x in np.linspace(a, b, int(1 / tol))]
    y = [func(x) for x in x]
    print(f"Настоящий минимум = {x[y.index(min(y))]:0.4f}")
    print("Дихотомия")
    print_stats(func, a, b, tol, dichotomy_minimization)
    print("Золотое сечение")
    print_stats(func, a, b, tol, golden_ratio_minimization)
    print("Фибоначчи")
    print_stats(func, a, b, tol, fibonacci_minimization)
    print("Парабола")
    print_stats(func, a, b, tol, parabolic_minimization)
    print("Брент")
    print_stats(func, a, b, tol, brent_minimization)


def multimodal():
    a = -10
    b = -2
    tol = 0.001
    x = [x for x in np.linspace(a, b, int(1 / tol))]
    y = [func(x) for x in x]
    print(f"Настоящий минимум = {x[y.index(min(y))]:0.4f}")
    print("Дихотомия")
    print_stats(func, a, b, tol, dichotomy_minimization)
    print("Золотое сечение")
    print_stats(func, a, b, tol, golden_ratio_minimization)
    print("Фибоначчи")
    print_stats(func, a, b, tol, fibonacci_minimization)
    print("Парабола")
    print_stats(func, a, b, tol, parabolic_minimization)
    print("Брент")
    print_stats(func, a, b, tol, brent_minimization)


unimodal()
multimodal()
