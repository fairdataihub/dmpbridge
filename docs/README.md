# DMPBridge documentation

[← Back to the main README](../README.md)

Start with the [main README](../README.md) if you just want to convert a PDF — it has the
install steps and the one command you need. These pages are for everything after that.

## I want to…

| I want to… | Go to |
|---|---|
| change the model, the extractor, or any flag | **[Configuration](configuration.md)** |
| run DMPBridge from Python instead of the shell | **[Configuration → From Python](configuration.md#from-python)** |
| run the whole sample set, not one PDF | **[Configuration → dmpbridge-wholedoc](configuration.md#dmpbridge-wholedoc--the-sample-set-all-four-stages)** |
| understand what happens between PDF and JSON | **[Pipeline](pipeline.md)** |
| know how a PDF is read, or fix a scanned PDF | **[Extraction](extraction.md)** |
| read the accuracy numbers, or score a new run | **[Scoring](scoring.md)** |
| fix something that broke | **[Troubleshooting](troubleshooting.md)** |
| edit the pipeline diagram | **[Editing the diagrams](DIAGRAMS.md)** |

## Something went wrong

The three that come up most often:

| What you see | Fix |
|---|---|
| `could not connect to ollama` | Ollama isn't running — start it with `ollama serve` |
| output reads well but doesn't match the PDF | the text layer is broken — [Extraction → broken PDFs](extraction.md#broken-or-scanned-pdfs) |
| a document is skipped and nothing runs | it already has output — [Troubleshooting → a document was skipped](troubleshooting.md#a-document-was-skipped) |

Everything else is in **[Troubleshooting](troubleshooting.md)**.
