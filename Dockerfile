FROM python:3.11-buster

WORKDIR /opt/app-root/
ENV PATH=/opt/app-root/bin:$PATH

# Create python virtual environment for installing required packages
RUN apt-get update && \
    apt-get install -y git nodejs npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/python -m venv /opt/app-root/ && \
    /opt/app-root/bin/pip install -U pip && \
    useradd -m -N -u 1001 -s /bin/bash -g 0 user && \
    chown -R 1001:0 /opt/app-root && \
    chmod -R og+rx /opt/app-root && \
    mkdir /opt/app-root/src

ENV H5DIR=/usr/local \
    H5VER=1.10.4 \
    HDF5_MD5SUM=cdf02e61f0d9920a7e7183aa0fb35429

# Install HDF5 and add library to the linker directory so they can be
# found by python libraries.
RUN set -ex \
    && wget -q https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$(echo $H5VER | cut -d. -f1-2)/hdf5-$H5VER/src/hdf5-$H5VER.tar.gz \
    && echo "$HDF5_MD5SUM hdf5-$H5VER.tar.gz" | md5sum -c - \
    && tar -xzvf hdf5-$H5VER.tar.gz \
    && cd hdf5-$H5VER \
    && ./configure --with-zlib --prefix=${H5DIR} --enable-shared \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ./hdf5-$H5VER.tar.gz ./hdf5-$H5VER \
    && echo "$H5DIR/lib" > /etc/ld.so.conf.d/local.conf \
    && ldconfig

COPY requirements.txt src/.

RUN pip install --no-cache-dir -r src/requirements.txt

COPY . src/.

WORKDIR /opt/app-root/src

RUN chown -R 1001:0 /opt/app-root

USER 1001
