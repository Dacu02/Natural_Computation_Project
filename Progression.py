import os
import time
import sys

EXP_PATH = os.path.join("experiments", "todo")
RESULT_PATH = os.path.join("experiments", "results")

def count_files(directory, extension):
    """Conta ricorsivamente i file con una certa estensione in una directory."""
    count = 0
    if not os.path.exists(directory):
        return 0
    
    for _, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Stampa una barra di avanzamento testuale."""
    if total == 0:
        percent = 0
    else:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    
    filled_length = int(length * iteration // total) if total > 0 else 0
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total and total > 0:
        print()

def main():
    print("Avvio monitoraggio esperimenti...")
    print(f"Monitorando: {EXP_PATH} (Pending) vs {RESULT_PATH} (Completed)")
    
    try:
        while True:
            
            pending_count = count_files(EXP_PATH, '.json')
            completed_count = count_files(RESULT_PATH, '.csv')
            
            total_experiments = pending_count + completed_count

            print("=== STATO AVANZAMENTO ESPERIMENTI ===\n")
            print(f"Totale Stimato:  {total_experiments}")
            print(f"Completati:      {completed_count}")
            print(f"Rimanenti:       {pending_count}")
            print("-" * 40)

            if total_experiments > 0:
                print_progress_bar(completed_count, total_experiments, prefix='Progress:', suffix='Completato', length=40)
            else:
                print("Nessun esperimento trovato nelle cartelle.")

            if pending_count == 0 and total_experiments > 0:
                print("\n\nUTTI GLI ESPERIMENTI SONO TERMINATI!")
                break
            
            # Aggiorna ogni 2 secondi
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMonitoraggio interrotto dall'utente.")

if __name__ == "__main__":
    main()