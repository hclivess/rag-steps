#!/bin/sh
# Double-click this file (file manager: "Run as a program") or start.desktop:
# it installs what is needed on the first start (into the .venv folder next to it),
# starts the web interface and opens it in the browser.  Close the window to stop it.
cd "$(dirname "$0")" || exit 1

fail() {
    echo
    echo "$1"
    printf "Press Enter to close. "
    read -r _
    exit 1
}

command -v python3 >/dev/null 2>&1 || fail "Python 3 was not found. Install it with your package manager (for example: sudo apt install python3 python3-venv)."

if [ ! -f .venv/installed.txt ]; then
    echo "First start: installing into the .venv folder here. This takes a minute or two."
    python3 -m venv .venv \
        && .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt \
        && echo ok > .venv/installed.txt
fi
[ -f .venv/installed.txt ] || fail "The installation failed. The messages above say why (on Debian and Ubuntu: sudo apt install python3-venv)."

.venv/bin/python -m rag serve "$@" || fail "The web interface stopped with an error."
