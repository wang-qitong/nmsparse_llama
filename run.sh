#!/bin/bash
port=33003
pwd=20050601Wqt!

mkdir -p /var/run/sshd
echo "root:${pwd}" | chpasswd
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "Port ${port}" >> /etc/ssh/sshd_config
service ssh start
sleep 60d
