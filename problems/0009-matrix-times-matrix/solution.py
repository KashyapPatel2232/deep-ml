def matrixmul(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]]:
    if not a or not b or len(a[0]) != len(b):
        return -1

    c = []
    for row in range(len(a)):
        out = []
        for col in range(len(b[0])):
            total = 0
            for k in range(len(b)):
                total += a[row][k] * b[k][col]
            out.append(total)
        c.append(out)
    return c