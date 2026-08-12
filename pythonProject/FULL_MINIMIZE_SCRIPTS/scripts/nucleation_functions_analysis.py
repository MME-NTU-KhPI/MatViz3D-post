import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Визначення функцій за формулами з картинки
def n_function(n, N_gr, N_p):
    # N(n) - Кумулятивна частка (0 до 1)
    return 0.5 * (1 + erf((n - N_p/2) / (N_gr**(1/3))))

def j_function(n, N_gr, N_p):
    # J(n) - Швидкість (кількість нових зерен на ітерацію)
    return (1 / (np.sqrt(N_gr * np.pi))) * np.exp(-((n - N_p/2)**2 / N_gr))

# Параметри аналізу
N_gr = 2197  # Загальна кількість зерен
n_range = np.linspace(0, 200, 1000) # Діапазон ітерацій
Np_values = [5, 10, 20, 50, 100, 150] # Варіанти параметра Np для порівняння

# --- РИСУНОК 1: Базова кумулятивна функція ---
plt.figure(figsize=(8, 5))
plt.plot(n_range, n_function(n_range, N_gr, 100), color='blue', lw=2)
plt.title("Cumulative Grain Count $N(n)$ (Base: $N_p=100$)")
plt.xlabel("Iteration (n)")
plt.ylabel("Fraction of total grains")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- РИСУНОК 2: Базова функція швидкості ---
plt.figure(figsize=(8, 5))
plt.plot(n_range, j_function(n_range, N_gr, 100), color='red', lw=2)
plt.title("Nucleation Rate $J(n)$ (Base: $N_p=100$)")
plt.xlabel("Iteration (n)")
plt.ylabel("Grains per iteration")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- РИСУНОК 3: Аналіз впливу Np на накопичення (N) ---
plt.figure(figsize=(8, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4', '#ff7f0e', '#2ca02c']
for i, np_val in enumerate(Np_values):
    plt.plot(n_range, n_function(n_range, N_gr, np_val),
             label=f'$N_p = {np_val}$ (Peak at $n={np_val/2}$)', color=colors[i])
plt.title("Analysis of $N(n)$ with varying $N_p$")
plt.xlabel("Iteration (n)")
plt.ylabel("Fraction of total grains")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- РИСУНОК 4: Аналіз впливу Np на інтенсивність (J) ---
plt.figure(figsize=(8, 5))
for i, np_val in enumerate(Np_values):
    plt.plot(n_range, j_function(n_range, N_gr, np_val),
             label=f'$N_p = {np_val}$', color=colors[i])
plt.title("Analysis of $J(n)$ with varying $N_p$")
plt.xlabel("Iteration (n)")
plt.ylabel("Grains per iteration")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()