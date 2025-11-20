from src.data_cleaning import clean_data
from src.normalization import normalize_column
from src.evaluation import evaluate_model
from src.pipeline import load_data, train_model, pipeline_evaluate
import pandas as pd

def run_ex1():
    print("\n=== Exercise 1: Data Cleaning ===")
    filepath = input("Enter CSV file path: ")
    df = pd.read_csv(filepath)
    cleaned = clean_data(df)
    print(cleaned)

def run_ex2():
    print("\n=== Exercise 2: Normalization ===")
    filepath = input("Enter CSV file path: ")
    column = input("Column to normalize: ")
    df = pd.read_csv(filepath)
    normalized = normalize_column(df, column)
    print(normalized)

def run_ex3():
    print("\n=== Exercise 3: Model Evaluation ===")
    y_true = [1, 0, 1]
    y_pred = [1, 0, 1]
    result = evaluate_model(y_true, y_pred)
    print(result)

def run_ex4():
    print("\n=== Exercise 4: GitHub Actions ===")
    print("Workflow file: .github/workflows/run-tests.yml")

def run_ex5():
    print("\n=== Exercise 5: ML Pipeline ===")
    filepath = input("Enter CSV file path: ")
    results = pipeline_evaluate(filepath)
    print(results)

def main():
    while True:
        print("\nSelect an option:")
        print("1. Exercise 1 (Data Cleaning)")
        print("2. Exercise 2 (Normalization)")
        print("3. Exercise 3 (Model Evaluation)")
        print("4. Exercise 4 (CI/CD GitHub Actions)")
        print("5. Exercise 5 (ML Pipeline)")
        print("0. Exit")

        choice = input("Enter option: ")

        if choice == "1":
            run_ex1()
        elif choice == "2":
            run_ex2()
        elif choice == "3":
            run_ex3()
        elif choice == "4":
            run_ex4()
        elif choice == "5":
            run_ex5()
        elif choice == "0":
            break
        else:
            print("Invalid option!")

if __name__ == "__main__":
    main()
