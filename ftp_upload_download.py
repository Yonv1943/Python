import ftplib

"""
Source: Yonv1943 2019-05-04
GitHub，使用FTP + pyftpdlib 上传下载远程服务器的文件
https://github.com/Yonv1943/Python/upload/master/Demo/ftp_upload_download.py

Python ftplib Tutorial
https://pythonprogramming.net/ftp-transfers-python-ftplib/

Read FTP files without writing them using Python.
https://stackoverflow.com/a/11209373/9293137
"""


def download_file(filename):
    localfile = open(filename, 'wb')
    ftp.retrbinary('RETR ' + filename, localfile.write, 1024)

    ftp.quit()
    localfile.close()


def upload_file(filename):
    ftp.storbinary('STOR ' + filename, open(filename, 'rb'))
    ftp.quit()


"""
run this command in server first:
sudo apt install python3-pyftpdlib
sudo python3 -m pyftpdlib  -i xxx.xxx.x.x -p 2121 -w
"""

ftp = ftplib.FTP(host='10.10.1.111')
print('Connect')

ftp.login(user='weit', passwd='weit2.71')
print('Login')

ftp.cwd('/home/')

upload_file('test.txt'), print('Upload')
download_file('test.txt'), print('Download')
