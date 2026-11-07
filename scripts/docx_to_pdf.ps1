# Converts a .docx to PDF with the installed Microsoft Word, to check how scripts/make_docx.py's
# output looks in Word itself.
#   powershell -NoProfile -File scripts\docx_to_pdf.ps1 <input.docx> <output.pdf>
param([Parameter(Mandatory)][string]$InputDocx, [Parameter(Mandatory)][string]$OutputPdf)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open((Resolve-Path $InputDocx).Path, $false, $true)   # read-only
    $doc.SaveAs([ref]([System.IO.Path]::GetFullPath($OutputPdf)), [ref]17)       # 17 = PDF
    "pages: " + $doc.ComputeStatistics(2)                                         # 2 = pages
    $doc.Close([ref]$false)
} finally {
    $word.Quit()
}
