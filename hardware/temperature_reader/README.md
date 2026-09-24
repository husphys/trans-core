# MEPI temperature reader

Source provenance: `/mnt/e/MEPI/read_temp.ino`  
Source SHA-256: `f1fa2c1e91dc3309fd512f832ab0f115944d24b31c33a2c49383351ac6f7615a`

The project copy preserves the program semantics; only CRLF line endings and
trailing horizontal whitespace were normalized for repository use. The source
path and hash above identify the byte-exact acquisition artifact.

Serial contract:

- port used by acquisition: `COM5`;
- baud: `9600`;
- command: ASCII `T` (`b"T"` in Python);
- response: two Celsius readings separated by a comma and terminated by a newline.

The Arduino identifiers/comments call the first MAX6675 channel `roomTemp` and
the second `coreTemp`. This is a software label, not proof of the installed
sensor positions. Physical tracing and experimenter verification established
that the installed first channel is TempCore and the installed second channel
is TempRoom. Therefore the authoritative acquisition API is:

```text
physical board field 1 = TempCore
physical board field 2 = TempRoom
software return        = (TempRoom, TempCore)
```

The source performs one `readCelsius()` call per channel for each `T` command.
It contains no averaging and no slope/offset calibration.
