#!/bin/bash
set -e

if [ -z "$GIT_USER_NAME" ] || [ -z "$GIT_USER_EMAIL" ]; then
    echo "GIT_USER_NAME or GIT_USER_EMAIL is not set. Please set them in your local environment."
    exit 1
fi

git config --global user.name "AdamKrysztopa"
git config --global user.email "krysztopa@gmail.com"
echo "Git global config updated: "AdamKrysztopa" <krysztopa@gmail.com>"
