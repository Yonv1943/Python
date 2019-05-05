from ftplib import FTP
import os


def run():
    ftp_url = 'ftp.nnvl.noaa.gov'
    dst_dir = os.path.join('f:', ftp_url)
    os.mkdir(dst_dir) if not os.path.exists(dst_dir) else None

    with FTP('ftp.nnvl.noaa.gov') as ftp:
        ftp.login()
        ftp.cwd('/GOES/GER/')

        dst_dirs = set([f for f in os.listdir(dst_dir) if f[-4:] == '.jpg'])
        src_dirs = set([f for f in ftp.nlst() if f[-4:] == '.jpg'])  # filter
        src_dirs = src_dirs - dst_dirs  # check local
        print("| dst:", len(dst_dirs))
        print("| src:", len(src_dirs))

        for i, download_file in enumerate(src_dirs):
            save_path = os.path.join(dst_dir, download_file)
            ftp.retrbinary('RETR % s' % download_file, open(save_path, 'wb').write)
            print("| %.4f %s" % (i/len(src_dirs), download_file))


if __name__ == '__main__':
    run()
