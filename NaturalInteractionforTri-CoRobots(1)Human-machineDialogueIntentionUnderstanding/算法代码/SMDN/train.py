import subprocess

# SNIPS, ATIS
for dataset in ['ATIS', 'SNIPS']:
    for proportion in ['25', '50', '75']:
        cmd_page = ['python', 'run_BiLSTM.py', dataset, proportion]
        subprocess.Popen(cmd_page, close_fds=True)
        cmd_page = ['python', 'run_BiLSTM-DOC.py', dataset, proportion]
        subprocess.Popen(cmd_page, close_fds=True)

# SwDA
for proportion in ['25', '50', '75']:
    cmd_page = ['python', 'run_HCNN.py', proportion]
    subprocess.Popen(cmd_page, close_fds=True)
    cmd_page = ['python', 'run_HCNN-DOC.py', proportion]
    subprocess.Popen(cmd_page, close_fds=True)