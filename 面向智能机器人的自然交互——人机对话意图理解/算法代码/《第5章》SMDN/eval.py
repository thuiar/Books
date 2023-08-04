import subprocess

for dataset in ['ATIS', 'SNIPS']:
    for proportion in ['25', '50', '75']:
        cmd_page = ['python', 'eval_BiLSTM.py', dataset, proportion]
        subprocess.Popen(cmd_page, close_fds=True)
        cmd_page = ['python', 'eval_BiLSTM-DOC.py', dataset, proportion]
        subprocess.Popen(cmd_page, close_fds=True)

for proportion in ['25', '50', '75']:
    cmd_page = ['python', 'eval_HCNN.py', proportion]
    subprocess.Popen(cmd_page, close_fds=True)
    cmd_page = ['python', 'eval_HCNN-DOC.py', proportion]
    subprocess.Popen(cmd_page, close_fds=True)