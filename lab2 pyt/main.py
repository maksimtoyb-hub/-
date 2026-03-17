import numpy as np           # Імпорт бібліотеки для роботи з масивами та математикою
import matplotlib.pyplot as plt # Імпорт модуля для побудови графіків
import csv                  # Імпорт модуля для роботи з файлами CSV
import os                   # Імпорт модуля для перевірки наявності файлів у системі

# ==========================================================
# 1. ПІДГОТОВКА ДАНИХ (Створення та читання CSV)
# ==========================================================
def prepare_data():
    # Визначаємо вхідні дані Варіанту 2: Залежність CPU (%) від RPS
    data = [
        ["RPS", "CPU"],
        [50, 20],
        [100, 35],
        [200, 60],
        [400, 110],
        [800, 210]
    ]
    # Відкриваємо файл 'data.csv' для запису
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f) # Створюємо об'єкт для запису CSV
        writer.writerows(data) # Записуємо всі рядки даних у файл

def read_data(filename):
    # Перевіряємо, чи існує файл; якщо ні — створюємо його функцією prepare_data
    if not os.path.exists(filename): prepare_data()
    x, y = [], [] # Списки для збереження значень колонок
    with open(filename, 'r') as f:
        reader = csv.DictReader(f) # Читаємо файл як словник (ключі - назви колонок)
        for row in reader:
            x.append(float(row['RPS'])) # Додаємо значення RPS у список x
            y.append(float(row['CPU'])) # Додаємо значення CPU у список y
    return np.array(x), np.array(y) # Повертаємо дані як масиви NumPy для обчислень

# ==========================================================
# 2. МАТЕМАТИЧНІ МЕТОДИ (Алгоритми інтерполяції)
# ==========================================================

def get_divided_diff_table(x, y):
    n = len(y) # Кількість точок даних
    table = np.zeros([n, n]) # Створюємо порожню матрицю розміром n x n
    table[:, 0] = y          # Перша колонка таблиці — це вихідні значення y
    for j in range(1, n):    # Обчислюємо розділені різниці від 1-го до (n-1)-го порядку
        for i in range(n - j):
            # Формула: (f(x1..xn) - f(x0..xn-1)) / (xn - x0)
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table # Повертаємо повну таблицю різниць

def newton_poly(x_nodes, coeffs, x):
    n = len(x_nodes) - 1 # Степінь полінома
    res = coeffs[n]      # Починаємо з останнього коефіцієнта (схема Горнера)
    for k in range(1, n + 1):
        # Рекурсивне обчислення значення полінома в точці x
        res = coeffs[n-k] + (x - x_nodes[n-k]) * res
    return res

def lagrange_poly(x_nodes, y_nodes, x):
    def basis(j): # Внутрішня функція для обчислення базисного полінома L_j(x)
        p = 1
        for i in range(len(x_nodes)):
            if i != j:
                # Формула базису: добуток (x - xi) / (xj - xi)
                p *= (x - x_nodes[i]) / (x_nodes[j] - x_nodes[i])
        return p
    # Підсумовуємо всі значення y, помножені на відповідні базисні поліноми
    return sum(y_nodes[j] * basis(j) for j in range(len(x_nodes)))

def factorial_poly(x_nodes, y_nodes, target_x):
    n = len(y_nodes) # Кількість вузлів
    h = np.mean(np.diff(x_nodes)) # Обчислюємо середній крок між вузлами (для рівномірної сітки)
    t = (target_x - x_nodes[0]) / h # Обчислюємо безрозмірний параметр t
    diffs = np.zeros((n, n)) # Матриця для скінченних різниць
    diffs[:, 0] = y_nodes
    for j in range(1, n): # Заповнюємо таблицю скінченних різниць (просте віднімання)
        for i in range(n - j):
            diffs[i, j] = diffs[i+1, j-1] - diffs[i, j-1]
    res = diffs[0, 0] # Початкове значення результату
    t_prod, fact = 1, 1 # Змінні для накопичення добутку t та факторіалу
    for k in range(1, n):
        t_prod *= (t - k + 1) # Обчислення t(t-1)(t-2)...
        fact *= k             # Обчислення k!
        res += (diffs[0, k] / fact) * t_prod # Додаємо черговий доданок формули Ньютона
    return res

# ==========================================================
# 3. ОСНОВНЕ ВИКОНАННЯ (Обчислення та вивід)
# ==========================================================

x_data, y_data = read_data('data.csv') # Читаємо дані з файлу
full_table = get_divided_diff_table(x_data, y_data) # Генеруємо таблицю розділених різниць
coeffs = full_table[0, :] # Коефіцієнти полінома — це перший рядок таблиці

# Вивід оформленої таблиці розділених різниць у консоль
print("\n" + "="*65)
print(f"{'ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ (ВАРІАНТ 2)':^65}")
print("="*65)
print(f"{'x':>5} | {'f(x)':>5} | {'1st diff':>10} | {'2nd diff':>10} | {'3rd diff':>10}")
print("-" * 65)
for i in range(len(x_data)):
    row = f"{x_data[i]:5.0f} | {y_data[i]:5.0f}" # Вивід аргументу та значення функції
    for j in range(1, len(x_data) - i):
        row += f" | {full_table[i][j]:10.6f}" # Додавання різниць вищих порядків
    print(row)

target = 600 # Точка, для якої робимо прогноз (600 RPS)
res_n = newton_poly(x_data, coeffs, target) # Прогноз методом Ньютона
res_f = factorial_poly(x_data, y_data, target) # Прогноз за формулою для рівних кроків

print("\n" + "-"*65)
print(f"Прогноз для {target} RPS (Ньютон): {res_n:.2f}%")
print(f"Прогноз для {target} RPS (Факторіальний): {res_f:.2f}%")
print("-" * 65)

# ==========================================================
# 4. ДОСЛІДНИЦЬКА ЧАСТИНА (Візуалізація результатів)
# ==========================================================

plt.figure(figsize=(16, 12)) # Створюємо полотно для 5-ти графіків

# --- ГРАФІК 1: Вплив кількості вузлів та ефект Рунге ---
plt.subplot(3, 2, 1) # Перша позиція в сітці графіків
x_fine = np.linspace(50, 800, 500) # Густа сітка точок для плавних ліній
for n in [5, 10, 20]: # Перевіряємо різні кількості вузлів
    x_n = np.linspace(50, 800, n)
    y_n = 0.25 * x_n + 10 + np.sin(x_n/40) * 7 # Функція з коливаннями
    c_n = get_divided_diff_table(x_n, y_n)[0, :]
    y_plot = [newton_poly(x_n, c_n, xi) for xi in x_fine]
    plt.plot(x_fine, y_plot, label=f'Вузлів n={n}') # Малюємо інтерполяційний поліном
plt.scatter(x_data, y_data, color='red', label='Дані Варіанту 2') # Позначаємо вихідні точки
plt.title("1. Вплив n та Ефект Рунге")
plt.legend(); plt.grid(True)

# --- ГРАФІК 2: Дослідження похибок (логарифмічна шкала) ---
plt.subplot(3, 2, 2)
for n in [5, 10, 20]:
    x_n = np.linspace(50, 800, n)
    y_n = 0.25 * x_n + 10 + np.sin(x_n/40) * 7
    c_n = get_divided_diff_table(x_n, y_n)[0, :]
    y_true = 0.25 * x_fine + 10 + np.sin(x_fine/40) * 7 # Істинне значення функції
    y_interp = np.array([newton_poly(x_n, c_n, xi) for xi in x_fine])
    plt.plot(x_fine, np.abs(y_true - y_interp), label=f'Похибка n={n}') # Графік модуля різниці
plt.yscale('log') # Використовуємо логарифмічну шкалу для Y
plt.title("2. Графік похибок (log)")
plt.legend(); plt.grid(True)

# --- ГРАФІК 3: Порівняння методів Ньютона та Лагранжа ---
plt.subplot(3, 2, 3)
y_newt = [newton_poly(x_data, coeffs, xi) for xi in x_fine] # Обчислення через Ньютона
y_lagr = [lagrange_poly(x_data, y_data, xi) for xi in x_fine] # Обчислення через Лагранжа
plt.plot(x_fine, y_newt, 'b-', lw=4, alpha=0.4, label='Ньютон') # Широка синя лінія
plt.plot(x_fine, y_lagr, 'r--', lw=1, label='Лагранж') # Тонка червона пунктирна лінія
plt.title("3. Ньютон vs Лагранж")
plt.legend(); plt.grid(True)

# --- ГРАФІК 4: Вплив величини кроку при однаковому інтервалі ---
plt.subplot(3, 2, 4)
for step in [150, 75]: # Перевіряємо два різні кроки дискретизації
    x_s = np.arange(50, 801, step)
    y_s = 0.25 * x_s + 10
    c_s = get_divided_diff_table(x_s, y_s)[0, :]
    plt.plot(x_fine, [newton_poly(x_s, c_s, xi) for xi in x_fine], label=f'Крок {step}')
plt.title("4. Вплив величини кроку")
plt.legend(); plt.grid(True)

# --- ГРАФІК 5: Фіксований крок, але зміна довжини інтервалу спостереження ---
plt.subplot(3, 2, 5)
fixed_step = 100
for end_val in [400, 600, 800]: # Як впливає розширення діапазону даних на прогноз
    x_i = np.arange(50, end_val + 1, fixed_step)
    y_i = 0.25 * x_i + 10
    c_i = get_divided_diff_table(x_i, y_i)[0, :]
    x_test = np.linspace(50, 800, 500)
    plt.plot(x_test, [newton_poly(x_i, c_i, xi) for xi in x_test], label=f'Інтервал [50, {end_val}]')
plt.axvline(x_data[-1], color='k', linestyle=':', alpha=0.3) # Малюємо лінію кінця даних
plt.title("5. Фіксований крок, змінний інтервал")
plt.legend(); plt.grid(True)

plt.tight_layout() # Оптимізуємо розташування графіків, щоб вони не накладалися
plt.show() # Відображаємо вікно з усіма побудованими графіками