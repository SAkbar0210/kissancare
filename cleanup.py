import os
import shutil
import glob

# Define directories to clean
LOGS_DIR = 'logs'
DATA_BACKUP_PATTERN = 'processed_data_backup_*'

def clean_directories():
    """Removes logs directory and data backup directories."""
    print(f"Checking for directories to clean...")

    # Clean logs directory
    if os.path.exists(LOGS_DIR):
        print(f"Removing existing logs directory: {LOGS_DIR}")
        try:
            shutil.rmtree(LOGS_DIR)
            print(f"{LOGS_DIR} removed successfully.")
        except OSError as e:
            print(f"Error removing {LOGS_DIR}: {e}")
    else:
        print(f"Logs directory not found: {LOGS_DIR}. Nothing to remove.")

    # Clean data backup directories
    backup_dirs = glob.glob(DATA_BACKUP_PATTERN)
    if backup_dirs:
        print(f"Found {len(backup_dirs)} data backup directories matching {DATA_BACKUP_PATTERN}. Removing...")
        for backup_dir in backup_dirs:
            print(f"Removing backup directory: {backup_dir}")
            try:
                shutil.rmtree(backup_dir)
                print(f"{backup_dir} removed successfully.")
            except OSError as e:
                print(f"Error removing {backup_dir}: {e}")
    else:
        print(f"No data backup directories found matching {DATA_BACKUP_PATTERN}. Nothing to remove.")

    print("Cleanup check complete.")

if __name__ == "__main__":
    clean_directories() 