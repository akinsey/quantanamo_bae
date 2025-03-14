# Prototype money printer

```bash
# get python version required for this project
asdf install
# reshim asdf
rm -rf ~/.asdf/shims && asdf reshim
# install packages
pip install -r requirements.txt
# run the project
python main.py
```

## Troubleshoot deps

reinstall deps when changing/updating python version

```bash
pip freeze | xargs pip uninstall -y
pip install -r requirements.txt
```
