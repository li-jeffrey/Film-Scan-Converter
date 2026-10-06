# Third-party notices

## NegPy

The negative conversion engine in `source/conversion/` (density metering, dye-crosstalk
unmixing, the H&D print curve, paper profiles, the slide transfer curve, and the bundled
crosstalk profiles and ICC profiles under `source/assets/`) is ported from
[NegPy](https://github.com/marcinz606/negpy) by marcinz606 and NegPy Contributors,
licensed under the GNU General Public License v3.0.

Ported files keep NegPy's module layout so upstream changes can be diffed. See
`docs/negpy-port-design.md` for the porting map and what was intentionally left out.
