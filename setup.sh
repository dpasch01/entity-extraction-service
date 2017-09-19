#!/bin/bash

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

cd service
chmod 777 build.sh
/bin/bash build.sh

