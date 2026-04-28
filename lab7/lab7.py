import random

def generate_and_save_data(n=100, exact_x=2.5):
    # Генеруємо матрицю A з діагональним переважанням
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        # Гарантуємо, що діагональний елемент більший за суму інших по модулю
        A[i][i] = row_sum + random.uniform(1, 5)

    # Обчислюємо вектор b (або f), виходячи з того, що всі x = 2.5
    b = [0.0] * n
    for i in range(n):
        b[i] = sum(A[i][j] * exact_x for j in range(n))

    # Записуємо у файли
    with open("matrix_A.txt", "w") as f_A:
        for row in A:
            f_A.write(" ".join(map(str, row)) + "\n")

    with open("vector_B.txt", "w") as f_B:
        for val in b:
            f_B.write(str(val) + "\n")
    print(f"Дані згенеровано та збережено (n={n}).")

def read_matrix(filename):
    with open(filename, "r") as f:
        return [[float(x) for x in line.split()] for line in f]


def read_vector(filename):
    with open(filename, "r") as f:
        return [float(line.strip()) for line in f]


def save_result_vector(filename, vector):
    """Зберігає результуючий вектор у текстовий файл"""
    with open(filename, "w") as f:
        for val in vector:
            f.write(str(val) + "\n")


def mat_vec_mult(A, x):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * x[j] for j in range(n))
    return res


def vector_norm(v):
    # Норма вектора (максимальне значення по модулю)
    return max(abs(x) for x in v)


def matrix_norm(A):
    # Норма матриці (максимальна сума модулів по рядках)
    return max(sum(abs(x) for x in row) for row in A)


# --- ЧАСТИНА 3: Ітераційні методи ---
def simple_iteration(A, b, eps=1e-14):
    n = len(A)
    x = [1.0] * n  # Початкове наближення
    tau = 1.0 / matrix_norm(A)  # Вибір параметра tau для збіжності
    iterations = 0

    while True:
        Ax = mat_vec_mult(A, x)
        x_new = [0.0] * n
        for i in range(n):
            # X^(k+1) = X^(k) - tau * (A*X^(k) - f)
            x_new[i] = x[i] - tau * (Ax[i] - b[i])

        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < eps:
            break
        x = x_new
        iterations += 1
    return x_new, iterations


def jacobi(A, b, eps=1e-14):
    n = len(A)
    x = [1.0] * n
    iterations = 0

    while True:
        x_new = [0.0] * n
        for i in range(n):
            # Сума для всіх елементів окрім діагонального
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < eps:
            break
        x = x_new
        iterations += 1
    return x_new, iterations


def seidel(A, b, eps=1e-14):
    n = len(A)
    x = [1.0] * n
    iterations = 0

    while True:
        x_new = list(x)  # Створюємо копію для поточної ітерації
        for i in range(n):
            # Використовуємо вже ОНОВЛЕНІ значення для j < i (метод Зейделя)
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Використовуємо СТАРІ значення для j > i
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < eps:
            break
        x = x_new
        iterations += 1
    return x_new, iterations


# --- ГОЛОВНИЙ БЛОК ВИКОНАННЯ ---
if __name__ == "__main__":
    # 1. Генеруємо та записуємо вихідні дані
    generate_and_save_data(n=100)

    # 2. Зчитуємо дані з файлів
    A = read_matrix("matrix_A.txt")
    b = read_vector("vector_B.txt")
    eps = 1e-14

    print("\nПочинаємо обчислення...")

    # Метод простої ітерації
    x_simple, iters_simple = simple_iteration(A, b, eps)
    print(f"Метод простої ітерації: {iters_simple} ітерацій")

    # Метод Якобі
    x_jacobi, iters_jacobi = jacobi(A, b, eps)
    print(f"Метод Якобі:           {iters_jacobi} ітерацій")

    # Метод Гауса-Зейделя
    x_seidel, iters_seidel = seidel(A, b, eps)
    print(f"Метод Гауса-Зейделя:   {iters_seidel} ітерацій")

    # 3. Зберігаємо результуючий вектор у файл
    save_result_vector("result_vector_X.txt", x_seidel)
    print("\nУспіх! Результуючий вектор X збережено у файл 'result_vector_X.txt'.")