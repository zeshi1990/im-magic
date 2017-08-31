import os
import random
import string

from paramiko import Transport, SFTPClient, RSAKey
import gdal

rand_str = lambda n: ''.join([random.choice(string.lowercase) for i in xrange(n)])


class RasterBrewer(object):

    def __init__(self, username, pkey_path):
        host = "glaser.berkeley.edu"
        port = 5441
        self.username = username
        self.rsakey = RSAKey.from_private_key_file(pkey_path)
        self.transport = Transport((host, port))
        self.transport.connect(username=self.username, pkey=self.rsakey)
        self.sftp = SFTPClient.from_transport(self.transport)

    def __del__(self):
        self.sftp.close()
        self.transport.close()

    def listdir(self, path, recursive=True):
        """
        List out the directory you are looking at
        :param path:
        :param recursive:
        :return:
        """
        def _listdir(p, i):
            try:
                ds = self.sftp.listdir(path=p)
            except IOError:
                return 0
            if i == "":
                new_i = "|---"
            else:
                new_i = " "*4 + i
            for temp_d in ds:
                if temp_d[0] == ".":
                    continue
                new_path = os.path.join(p, temp_d)
                print "{0}{1}".format(i, temp_d)
                _listdir(new_path, new_i)

        indent = ""
        if recursive:
            _listdir(path, i=indent)
        else:
            try:
                dirs = self.sftp.listdir(path=path)
            except IOError:
                print "{0} is not a directory".format(path)
                return
            for temp_dir in dirs:
                print "{0}{1}".format(indent, temp_dir)

    def fetch_raster(self, src_path):
        dst_path = "/tmp/{0}".format(os.path.basename(src_path))
        self.sftp.get(src_path, dst_path)
        ds = gdal.Open(dst_path)
        os.remove(dst_path)
        return ds

