import os
import random
import string

from paramiko import Transport, SFTPClient, RSAKey
import gdal

rand_str = lambda n: ''.join([random.choice(string.lowercase) for i in xrange(n)])


class RasterBrewer(object):

    def __init__(self, username, pkey_path):
        """
        Intialize the RasterBrewer instance

        Parameters
        ----------

        username : str, username of your glasercompserver account

        pkey_path : str, private key directory on your local machine

        """
        host = "glaser.berkeley.edu"
        port = 5441
        self._username = username
        self._rsakey = RSAKey.from_private_key_file(pkey_path)
        self._transport = Transport((host, port))
        self._transport.connect(username=self._username, pkey=self._rsakey)
        self._sftp = SFTPClient.from_transport(self._transport)

    def __del__(self):
        self._sftp.close()
        self._transport.close()

    def listdir(self, path, recursive=True):
        """
        List out the directory specified

        Parameters
        ----------

        path : str, absolute path on glasercompserver

        recursive : bool, If True show the path tree recursively, else show files and directories in the folder

        Returns
        -------

        None
        """
        def _listdir(p, i):
            try:
                ds = self._sftp.listdir(path=p)
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
                dirs = self._sftp.listdir(path=path)
            except IOError:
                print "{0} is not a directory".format(path)
                return
            for temp_dir in dirs:
                print "{0}{1}".format(indent, temp_dir)

    def mkdir(self, path):
        self._sftp.mkdir(path)
        return 0

    def listdir(self, path):
        return self._sftp.listdir(path)

    def isdir(self, path):
        parent_dir = os.path.abspath(path).rsplit(os.path.sep, 1)[0]
        l_path = self.listdir(parent_dir)
        return os.path.basename(path) in l_path

    def fetch_raster(self, src_path):
        """
        Fetch a raster file from the remote directory and open it in gdal

        Parameters
        ----------

        src_path : str, path of the source file on the remote

        Returns
        -------

        ds : gdal.Dataset

        """
        dst_path = "/tmp/{0}".format(os.path.basename(src_path))
        self._sftp.get(src_path, dst_path)
        ds = gdal.Open(dst_path)
        os.remove(dst_path)
        return ds

    def upload_raster(self, src_path, dst_path):
        """
        upload a raster file to the remote directory
        :param src_path:
        :param dst_path:
        :return:
        """
        self._sftp.put(src_path, dst_path)
        return 0
