import sys
import os
import time
import tarfile

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse
    import urllib

class ImageNetDownloader:
    def __init__(self, data_path='../data/raw/'):
        self.host = 'http://www.image-net.org'
        self.data_path = data_path

    def download_file(self, url, desc=None, renamed_file=None):
        u = urllib2.urlopen(url)

        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = self.data_path + os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'

        if not renamed_file is None:
            filename = renamed_file

        if desc:
            filename = os.path.join(desc, filename)

        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            print("Downloading: {0} Bytes: {1}".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)

        return filename

    def extractTarfile(self, filename):
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()


    def mkWnidDir(self, wnid, desc):
    	tt = self.data_path + desc + '_' + wnid
        if not os.path.exists(tt):
            os.mkdir(tt)
        return os.path.abspath(tt)

    def downloadOriginalImages(self, wnid, username, accesskey, desc):
        download_url = 'http://www.image-net.org/download/synset?wnid=%s&username=%s&accesskey=%s&release=latest&src=stanford' % (wnid, username, accesskey)
        try:
             download_file = self.download_file(download_url, self.mkWnidDir(wnid, desc), wnid + '_original_images.tar')
        except Exception, erro:
            print 'Fail to download : ' + download_url
        currentDir = os.getcwd()
        extracted_folder = self.mkWnidDir(wnid, desc)
        if not os.path.exists(extracted_folder):
            os.mkdir(extracted_folder)
        extracted_folder = os.path.join(extracted_folder, wnid + '_original_images')
        if not os.path.exists(extracted_folder):
            os.mkdir(extracted_folder)
        os.chdir(extracted_folder)
        self.extractTarfile(download_file)
        os.chdir(currentDir)
        print 'Extract images to ' + extracted_folder

