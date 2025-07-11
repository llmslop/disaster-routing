# Data processing scripts

These scripts are mainly designed for Unix/Linux platforms, so it is not cross
platform. Here is an overview of what these scripts are for.

The philosophy of these data processing scripts is **avoid saving intermediate
data to disk**. The main way to avoid such hassle is by piping, but there are
cases when temporaries is unavoidable (e.g. Excel spreadsheets). In these cases,
`open_temp.sh` can be used.

Python scripts can be run in two ways:

- Via `uv`: this handles dev-dependencies automatically, but you'll have to type
  `uv run` manually, which could bloat the command.
- Via shebang: this will run the default `python3` executor, which might not
  have the necessary dependencies. You will have to either: activate the uv
  virtualenv via `source .venv/bin/activate`, or make sure that you have all
  necessary dev dependencies installed.

TL;DR here is the current primary usage of these scripts:

```sh
source .venv/bin/activate # these scripts needs uv dev dependencies
scripts/collect_logs.py zip://output.log.zip    \
    | scripts/process_results.py                \
    | scripts/open_temp.sh .xlsx
```

## `open.sh` and `open_temp.sh`

We assume you have a
[resource opener](https://wiki.archlinux.org/title/Default_applications#Resource_openers)
installed. Make your own `open.sh` (this file is gitignored by default), such as:

```sh
#!/bin/sh
gio open $@
# or xdg-open $@
# etc.
```

`open_temp.sh` is a handy script to open _temporary_ files. When some content is
piped to this script, it creates a temporary file via `mktemp`, and use the
`open.sh` script to open this file.

## `collect_logs.py`

This script collect log from a virtual filesystem (a directory, a ZIP file,
or anything really):

```sh
# from the multirun directory
scripts/collect_logs.py multirun

# from a zip file (handy since no manual extracting step is required)
scripts/collect_logs.py zip://output.log.zip
```

The output of this file is a JSON array containing all log messages and relevant
configs. This can be piped into a log processing script to process the logged
results.

## `procces_results.py`

This script processes log messages from stdin (in the format outputted from
`collect_logs.py`) and output an XLSX file to stdout. This result can be piped
to `open_temp.sh` to open this file in an editor (e.g. LibreOffice).
