import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. ОТРИМАННЯ ДАНИХ ЧЕРЕЗ API
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

try:
    response = requests.get(url, timeout=20)
    results = response.json()["results"]
except Exception as e:
    print(f"Помилка з'єднання: {e}")
    exit()


# Математичні функції
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def solve_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)
    A, B, C, D = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    B[0] = B[n] = 1
    for i in range(1, n):
        A[i], B[i], C[i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        D[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.zeros(n + 1);
    alpha = np.zeros(n);
    beta = np.zeros(n)
    for i in range(n):
        m = A[i] * alpha[i - 1] + B[i]
        alpha[i], beta[i] = -C[i] / m, (D[i] - A[i] * beta[i - 1]) / m
    c[n] = (D[n] - A[n] * beta[n - 1]) / (A[n] * alpha[n - 1] + B[n])
    for i in range(n - 1, -1, -1): c[i] = alpha[i] * c[i + 1] + beta[i]
    a = y[:-1]
    d = (c[1:] - c[:-1]) / (3 * h)
    b = (y[1:] - y[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
    return a, b, c[:-1], d


# Обробка даних
all_elev = np.array([p['elevation'] for p in results])
all_dist = [0]
for i in range(1, len(results)):
    d = haversine(results[i - 1]['latitude'], results[i - 1]['longitude'], results[i]['latitude'],
                  results[i]['longitude'])
    all_dist.append(all_dist[-1] + d)
all_dist = np.array(all_dist)

# ВІЗУАЛІЗАЦІЯ
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Порівняльний аналіз сплайнів та аналіз градієнта", fontsize=16)

node_counts = [10, 15, 20]
errors = {}

for i, count in enumerate(node_counts):
    ax = plt.subplot(2, 3, i + 1)
    indices = np.linspace(0, len(all_dist) - 1, count, dtype=int)
    x_n, y_n = all_dist[indices], all_elev[indices]
    a, b, c, d = solve_spline(x_n, y_n)

    x_f = np.linspace(x_n[0], x_n[-1], 300)
    y_f = []
    for v in x_f:
        idx = max(0, min(np.searchsorted(x_n, v) - 1, len(a) - 1))
        dt = v - x_n[idx]
        y_f.append(a[idx] + b[idx] * dt + c[idx] * dt ** 2 + d[idx] * dt ** 3)

    ax.plot(x_f, y_f, label=f'Сплайн {count}')
    ax.scatter(x_n, y_n, color='red', s=15)
    ax.set_title(f"Модель: {count} вузлів")
    ax.set_xlabel("Відстань (м)")
    ax.set_ylabel("Висота (м)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Розрахунок похибки
    y_interp = []
    for val in all_dist:
        idx = max(0, min(np.searchsorted(x_n, val) - 1, len(a) - 1))
        dt = val - x_n[idx]
        y_interp.append(a[idx] + b[idx] * dt + c[idx] * dt ** 2 + d[idx] * dt ** 3)
    errors[count] = np.abs(all_elev - np.array(y_interp))

# 4. ОСНОВНИЙ ГРАФІК ТА АНАЛІЗ ГРАДІЄНТА
ax_main = plt.subplot(2, 3, 4)
a_m, b_m, c_m, d_m = solve_spline(all_dist, all_elev)
x_m = np.linspace(all_dist[0], all_dist[-1], 500)
y_m = []
for v in x_m:
    idx = max(0, min(np.searchsorted(all_dist, v) - 1, len(a_m) - 1))
    dt = v - all_dist[idx]
    y_m.append(a_m[idx] + b_m[idx] * dt + c_m[idx] * dt ** 2 + d_m[idx] * dt ** 3)

ax_main.plot(x_m, y_m, 'g-', linewidth=2, label='Основний графік')
ax_main.scatter(all_dist, all_elev, color='black', s=10)
ax_main.set_title("ОСНОВНИЙ МАРШРУТ")
ax_main.set_xlabel("Відстань (м)")
ax_main.set_ylabel("Висота (м)")
ax_main.set_facecolor('#f0f8ff')
ax_main.grid(True)
ax_main.legend()

# 5. ГРАФІК ПОХИБКИ
ax_err = plt.subplot(2, 3, 5)
for count, err in errors.items():
    ax_err.plot(all_dist, err, label=f'N={count}')
ax_err.set_title("ГРАФІК ПОХИБКИ")
ax_err.set_xlabel("Відстань (м)")
ax_err.set_ylabel("Похибка (м)")
ax_err.grid(True)
ax_err.legend()

# ВИКОНАННЯ ДОДАТКОВИХ ЗАВДАНЬ
print("\n" + "=" * 40)
print("1. ХАРАКТЕРИСТИКИ МАРШРУТУ")
print(f"Загальна довжина маршруту (м): {all_dist[-1]:.2f}")

total_ascent = sum(max(all_elev[i] - all_elev[i - 1], 0) for i in range(1, len(all_elev)))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")

total_descent = sum(max(all_elev[i - 1] - all_elev[i], 0) for i in range(1, len(all_elev)))
print(f"Сумарний спуск (м): {total_descent:.2f}")

grad_full = np.gradient(y_m, x_m[1] - x_m[0]) * 100
print("\n2. АНАЛІЗ ГРАДІЄНТА")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")
print("=" * 40)

print("\n3. ЕНЕРГЕТИЧНІ ВИТРАТИ")
mass, g = 80, 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()