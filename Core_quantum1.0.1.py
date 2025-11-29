import numpy as np
import matplotlib.pyplot as plt


class QuantumSimulator:
    def __init__(self, mode="basic"):
        """
        Инициализация симулятора.
        mode:
          - "basic" — 6 кубитов (H, X, CNOT)
          - "shor" — 10 кубитов (упрощённый Шора)
        """
        self.mode = mode
        if mode == "basic":
            self.n_qubits = 6
            print("Режим: базовый (6 кубитов)")
        elif mode == "shor":
            self.n_qubits = 10
            print("Режим: упрощённый Шора (10 кубитов)")
        else:
            raise ValueError("Режим должен быть 'basic' или 'shor'")

        # Начальное состояние |00...0⟩
        self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self.state[0] = 1

    def apply_hadamard(self, qubit):
        """Применить гейт Хадамара к указанному кубиту."""
        if not (0 <= qubit < self.n_qubits):
            print(f"Ошибка: кубит должен быть от 0 до {self.n_qubits-1}")
            return
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self._apply_single_qubit_gate(H, qubit)

    def apply_paulix(self, qubit):
        """Применить гейт Паули-X к указанному кубиту."""
        if not (0 <= qubit < self.n_qubits):
            print(f"Ошибка: кубит должен быть от 0 до {self.n_qubits-1}")
            return
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._apply_single_qubit_gate(X, qubit)

    def apply_cnot(self, control, target):
        """Применить CNOT (только в режиме basic)."""
        if self.mode != "basic":
            print("CNOT доступен только в базовом режиме (6 кубитов)")
            return
        if not (0 <= control < 6 and 0 <= target < 6):
            print("Ошибка: control и target должны быть от 0 до 5")
            return
        if control == target:
            print("Ошибка: control и target не могут совпадать")
            return

        # Создаём новый вектор состояний (чтобы не портить исходные амплитуды при итерации)
        new_state = np.zeros_like(self.state)

        for idx in range(2**6):
            # Если control-кубит в состоянии |1⟩, инвертируем target
            if (idx >> control) & 1:
                idx_flipped = idx ^ (1 << target)  # Инвертируем target-кубит
                new_state[idx_flipped] += self.state[idx]
            else:
                # Если control=0, состояние остаётся без изменений
                new_state[idx] += self.state[idx]

        self.state = new_state

    def _apply_single_qubit_gate(self, gate, qubit):
        """Применить однокубитный гейт (вспомогательный метод)."""
        state_copy = self.state.copy()
        stride = 2 ** qubit
        for idx in range(0, 2**self.n_qubits, 2 * stride):
            for offset in range(stride):
                i0 = idx + offset
                i1 = i0 + stride
                # Применяем гейт к паре амплитуд
                self.state[i0] = (gate[0, 0] * state_copy[i0] +
                                  gate[0, 1] * state_copy[i1])
                self.state[i1] = (gate[1, 0] * state_copy[i0] +
                                  gate[1, 1] * state_copy[i1])

    def reset(self):
        """Сбросить состояние в |00...0⟩."""
        self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self.state[0] = 1

    def get_state(self):
        """Вернуть текущее состояние."""
        return self.state

    def print_state(self, threshold=1e-6):
        """Вывести состояния с вероятностью > threshold."""
        probs = np.abs(self.state)**2
        print(f"Состояния с вероятностью > {threshold:.0e}:")
        found = False
        for i, prob in enumerate(probs):
            if prob > threshold:
                print(f"|{i:0{self.n_qubits}b}⟩: {prob:.6f}")
                found = True
        if not found:
            print("Нет состояний с вероятностью выше порога.")

    def visualize_state(self):
        """Визуализировать вероятности состояний (p > 0.001)."""
        probabilities = np.abs(self.state) ** 2
        mask = probabilities > 0.001
        if not np.any(mask):
            print("Нет состояний с вероятностью > 0.001")
            return

        labels = [f"|{i:0{self.n_qubits}b}⟩" for i in np.where(mask)[0]]
        probs_filtered = probabilities[mask]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

        plt.figure(figsize=(12, 6))
        plt.bar(labels, probs_filtered, color=colors)
        plt.title(
            f"Вероятности состояний (p > 0.001), {self.n_qubits} кубитов")
        plt.xlabel("Состояния")
        plt.ylabel("Вероятность")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_simplified_shor(self):
        """Упрощённый алгоритм Шора (только в режиме shor)."""
        if self.mode != "shor":
            print("Алгоритм Шора доступен только в режиме 'shor' (10 кубитов)")
            return

        print("Запуск упрощённого алгоритма Шора (10 кубитов)...")
        # Пример: применяем H ко всем кубитам
        for q in range(10):
            self.apply_hadamard(q)
        print("Упрощённый Шора: применён H ко всем 10 кубитам.")

    def help(self):
        """Вывести список доступных команд."""
        print("Доступные команды:")
        print("  h <qubit>      - Гейт Хадамара (0 ≤ qubit < n_qubits)")
        print("  x <qubit>      - Гейт Паули-X (0 ≤ qubit < n_qubits)")
        print("  cnot <c> <t>  - CNOT: control <c>, target <t> (только 6 кубитов)")
        print("  shor          - Упрощённый алгоритм Шора (только 10 кубитов)")
        print("  reset         - Сбросить состояние в |00...0⟩")
        print("  state         - Показать полное состояние")
        print("  print         - Вывести состояния (p > 1e-6)")
        print("  visualize     - Визуализировать вероятности (p > 0.001)")
        print("  help          - Эта справка")
        print("  exit          - Выход")
        print("  mode <basic|shor> - Переключить режим (6 или 10 кубитов)")

    def change_mode(self, new_mode):
        """Переключить режим симулятора."""
        if new_mode not in ["basic", "shor"]:
            print("Ошибка: режим должен быть 'basic' или 'shor'")
            return

        self.mode = new_mode
        if new_mode == "basic":
            self.n_qubits = 6
            print("Режим изменён на базовый (6 кубитов)")
        elif new_mode == "shor":
            self.n_qubits = 10
            print("Режим изменён на упрощённый Шора (10 кубитов)")

        # Сброс состояния
        self.reset()


def main():
    print("Выберите режим:")
    print("  1 — Базовый (6 кубитов: H, X, CNOT)")
    print("  2 — Упрощённый Шора (10 кубитов)")
    choice = input("Введите 1 или 2: ").strip()

    if choice == "1":
        sim = QuantumSimulator(mode="basic")
    elif choice == "2":
        sim = QuantumSimulator(mode="shor")
    else:
        print("Неверный выбор, запуск в базовом режиме.")
        sim = QuantumSimulator(mode="basic")

    running = True
    while running:
        print("\nТекущее состояние:", sim.get_state())
        user_input = input("Введите команду (help для справки): ").strip()

        if not user_input:
            continue  # Пропустить пустую команду

        elif user_input.lower() == 'help':
            sim.help()

        elif user_input.startswith('h '):
            try:
                qubit = int(user_input.split()[1])
                sim.apply_hadamard(qubit)
            except (ValueError, IndexError):
                print("Ошибка: укажите номер кубита (целое число).")

        elif user_input.startswith('x '):
            try:
                qubit = int(user_input.split()[1])
                sim.apply_paulix(qubit)
            except (ValueError, IndexError):
                print("Ошибка: укажите номер кубита (целое число).")

        elif user_input.startswith('cnot '):
            try:
                parts = user_input.split()
                control = int(parts[1])
                target = int(parts[2])
                sim.apply_cnot(control, target)
            except (ValueError, IndexError):
                print("Ошибка: формат 'cnot <control> <target>'.")

        elif user_input == 'shor':
            sim.run_simplified_shor()

        elif user_input == 'reset':
            sim.reset()
            print("Состояние сброшено в |00...0⟩.")

        elif user_input == 'state':
            print("Полное состояние системы:")
            print(sim.get_state())

        elif user_input == 'print':
            sim.print_state()

        elif user_input == 'visualize':
            sim.visualize_state()

        elif user_input.startswith('mode '):
            try:
                new_mode = user_input.split()[1]
                if new_mode not in ["basic", "shor"]:
                    print("Ошибка: режим должен быть 'basic' или 'shor'")
                    continue
                sim.change_mode(new_mode)
            except IndexError:
                print("Ошибка: укажите режим ('basic' или 'shor').")

        elif user_input == 'exit':
            print("До свидания!")
            running = False

        else:
            print("Неверная команда. Введите 'help' для списка команд.")


if __name__ == "__main__":
    main()
