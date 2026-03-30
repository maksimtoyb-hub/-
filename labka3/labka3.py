import os
import csv
import matplotlib.pyplot as plt

# Налаштовуємо приємний оку візуал для графіків
plt.style.use('seaborn-v0_8-muted')


# Функція для "добудовування" значень y між відомими точками (лінійна інтерполяція)
def get_y_true(x_val, x_nodes, y_nodes):
    for i in range(len(x_nodes) - 1):
        # Перевіряємо, чи входить наш x у поточний відрізок між вузлами
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            # Рахуємо y за формулою прямої, що проходить через ці дві точки
            return y_nodes[i] + (y_nodes[i + 1] - y_nodes[i]) * (x_val - x_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
    return y_nodes[-1]  # якщо x поза межами, віддаємо останнє значення


# Читаємо дані з нашого CSV файлу
def read_data(filename):
    x, y = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Просто пропускаємо заголовок таблиці
        for row in reader:
            x.append(float(row[0]))  # Записуємо перший стовпчик як x
            y.append(float(row[1]))  # Другий стовпчик як y (значення функції)
    return x, y


# Будуємо матрицю коефіцієнтів для системи нормальних рівнянь
def form_matrix(x, m):
    # Створюємо порожню квадратну матрицю розміром (m+1)x(m+1)
    a = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            # Елементи матриці — це суми іксів у відповідних ступенях
            a[i][j] = sum(xi ** (i + j) for xi in x)
    return a


# Формуємо праву частину (вектор b) для розв'язання системи
def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        # Кожен елемент — це сума добутків y на x у ступені i
        b[i] = sum(y[k] * (x[k] ** i) for k in range(len(x)))
    return b


# Реалізація методу Гаусса з вибором найбільшого елемента (щоб не було помилок при діленні)
def gauss_solve(a, b):
    n = len(a)
    a_copy = [row[:] for row in a]  # Копіюємо дані, щоб не затерти оригінал
    b_copy = b[:]

    for k in range(n - 1):
        # Шукаємо рядок, де елемент у поточному стовпці найбільший
        max_row = k
        for i in range(k + 1, n):
            if abs(a_copy[i][k]) > abs(a_copy[max_row][k]):
                max_row = i

        # Міняємо поточний рядок з тим, де знайшли максимум
        a_copy[k], a_copy[max_row] = a_copy[max_row], a_copy[k]
        b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]

        # Процес виключення: робимо нулі під головною діагоналлю
        for i in range(k + 1, n):
            if a_copy[k][k] == 0: continue
            factor = a_copy[i][k] / a_copy[k][k]
            for j in range(k, n):
                a_copy[i][j] -= factor * a_copy[k][j]
            b_copy[i] -= factor * b_copy[k]

    # Зворотний хід: виражаємо невідомі коефіцієнти один за одним
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(a_copy[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b_copy[i] - s) / a_copy[i][i]
    return x_sol


# Функція для розрахунку значень нашого полінома в заданих точках
def polynomial(x_vals, coef):
    # Підставляємо коефіцієнти у формулу a0 + a1*x + a2*x^2 ...
    return [sum(coef[i] * (xv ** i) for i in range(len(coef))) for xv in x_vals]


# Рахуємо середню помилку (дисперсію) апроксимації
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    # Сума квадратів різниць між реальним значенням і отриманим
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n


# Основна логіка виконання
current_dir = os.path.dirname(os.path.abspath(__file__))  # Дізнаємось, де лежить скрипт
data_path = os.path.join(current_dir, 'data.csv')  # Склеюємо шлях до файлу з даними

x, y = read_data(data_path)  # Завантажуємо дані

variances = []  # Сюди будемо складати помилки для різних ступенів
max_degree = 10  # Перевіримо поліноми від 1-го до 10-го ступеня
n_nodes = len(x)

# Проходимо циклом по ступенях полінома
for m in range(1, max_degree + 1):
    a_mat = form_matrix(x, m)  # Матриця системи
    b_vec = form_vector(x, y, m)  # Права частина
    coef = gauss_solve(a_mat, b_vec)  # Знаходимо коефіцієнти
    y_approx = polynomial(x, coef)  # Рахуємо точки за моделлю
    var = calculate_variance(y, y_approx)  # Оцінюємо якість
    variances.append(var)  # Зберігаємо результат

# Вибираємо той ступінь m, де помилка була найменшою
optimal_m = variances.index(min(variances)) + 1

# Будуємо фінальну модель на основі знайденого ідеального ступеня
a_opt = form_matrix(x, optimal_m)
b_opt = form_vector(x, y, optimal_m)
coef_opt = gauss_solve(a_opt, b_opt)

# Створюємо дуже густу сітку точок для малювання плавної кривої
x_smooth = [x[0] + i * (x[-1] - x[0]) / 200 for i in range(201)]
y_smooth = polynomial(x_smooth, coef_opt)

# Прогнозуємо значення на три місяці вперед (25, 26, 27)
x_future = [25, 26, 27]
y_future = polynomial(x_future, coef_opt)

# Робимо кроки для детального аналізу похибки всередині інтервалу
h1 = (x[-1] - x[0]) / (20 * n_nodes)
x_err = []
curr_x = x[0]
while curr_x <= x[-1]:
    x_err.append(curr_x)
    curr_x += h1

# --- РОБОТА З ГРАФІКАМИ ---

# Малюємо перший графік: як змінюється помилка від ступеня полінома
plt.figure(1, figsize=(10, 5))
plt.plot(range(1, max_degree + 1), variances, color='#4A90E2', marker='o', markersize=8, linewidth=2)
plt.axvline(x=optimal_m, color='#E94E77', linestyle='--', label=f'Оптимальне m={optimal_m}')
plt.fill_between(range(1, max_degree + 1), variances, color='#4A90E2', alpha=0.1)
plt.title("Залежність дисперсії від ступеня полінома", fontsize=14, pad=15)
plt.xlabel("Ступінь (m)", fontweight='bold')
plt.ylabel("Дисперсія", fontweight='bold')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Малюємо другий графік: фактичні точки, лінія моделі та точки прогнозу
plt.figure(2, figsize=(10, 6))
plt.scatter(x, y, color='#333333', alpha=0.5, label='Фактичні дані', s=40)
plt.plot(x_smooth, y_smooth, color='#4A90E2', linewidth=2.5, label=f'Апроксимація (m={optimal_m})')
plt.plot(x_future, y_future, color='#E94E77', linestyle='--', marker='s', markersize=6, linewidth=2, label='Прогноз')
plt.title("Апроксимація та прогноз температури", fontsize=14, pad=15)
plt.xlabel("Місяць", fontweight='bold')
plt.ylabel("Температура", fontweight='bold')
plt.legend(facecolor='white', framealpha=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Малюємо третій графік: показуємо розподіл похибки вздовж усього графіка
plt.figure(3, figsize=(10, 6))
for m in range(1, max_degree + 1):
    a_mat = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    c_m = gauss_solve(a_mat, b_vec)
    y_approx_err = polynomial(x_err, c_m)
    y_true_err = [get_y_true(xv, x, y) for xv in x_err]
    error_vals = [abs(y_true_err[i] - y_approx_err[i]) for i in range(len(x_err))]

    # Виділяємо червоним саме оптимальний варіант, інші робимо напівпрозорими
    if m == optimal_m:
        plt.plot(x_err, error_vals, color='#D0021B', linewidth=3, label=f'm={m} (Оптимальна)', zorder=10)
    else:
        plt.plot(x_err, error_vals, alpha=0.2, color='#9B9B9B')

plt.title("Розподіл абсолютної похибки", fontsize=14, pad=15)
plt.xlabel("Місяць", fontweight='bold')
plt.ylabel("Абсолютна похибка", fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, linestyle='-', alpha=0.3)
plt.tight_layout()

plt.show()