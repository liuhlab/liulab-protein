# Changelog

Every change worth knowing about, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers are CalVer tags
of the form `vYYYY.M.PATCH`, and the tag is where the version comes from — nothing here
sets one.

## [Unreleased]

### Added

- **Alignments, built here at last.** `MSA` holds one, and reads and writes A3M.
  `Protein.msa(db)` searches a database and hands back an alignment in memory, the way
  `Protein.search(db)` hands back a table of hits. `align(sequences, query=...)` runs MUSCLE
  over a set you already hold, and anchors the result on the query you name. Both ways in are
  also `protein msa` on the command line.

  **Case is the whole point of an A3M.** A lowercase letter is an insertion and takes no
  column; an uppercase one holds a column. An alignment that has been put in upper case has
  lost the one thing that made it an A3M, so `MSA` holds plain text and never biotite's
  `Alignment`. Headers survive byte for byte, because a `key=` field in a header is what pairs
  the chains of a complex. A leading `#` line survives for the same reason.

  A ragged alignment is refused when it is built, not two steps later. `muscle` joins `mmseqs`
  and `foldseek` as a required tool, and `doctor()` reports all three.

- **Structure prediction, with ESMFold2.** `ESMFold2` loads the weights once and keeps them.
  `fold()` takes a `FoldingRequest` and a directory you name, and hands back a `Structure` — so
  everything already built on that class works on a prediction with no conversion. Protein, DNA
  and RNA chains fold together. `protein fold` does the same from the command line.

  **A prediction is named for the molecule, not for the model.** A name you give wins; failing
  that the accession; failing that a short hash of the sequence. Fold the same thing twice and
  the second call returns the first answer. Fold a different sequence under a name already
  taken and it raises, unless you pass `overwrite=`. The stored sequence is read back off the
  residues, so that check still holds a day later.

  Per-residue confidence rides in the B-factor column, so every viewer colours by it. For a
  complex, the pairwise matrix is written beside the file. `load_esmc=False` now warns: it
  returns a file of the right length holding the wrong structure, and used to say nothing.

  **The card's limit is yours to hit.** There is no length cap and no memory arithmetic here. A
  fold that does not fit raises whatever the GPU raises.

- **Sparse features from an embedding.** `SAE(slug)` loads a sparse autoencoder; `encode()`
  turns an `Embedding` into a `SaeActivation`. That is a peer of `Embedding` and not one of
  them: it holds feature numbers and their values, not a dense row per residue.

  **A wrong pairing is now an error instead of plausible numbers.** Give an autoencoder the
  wrong model's embedding, or the right model's wrong layer, and the shapes agree and numbers
  come out. Only the quality of the rebuild says otherwise, and nothing looked at it. `encode`
  checks the model, the layer and the width, and says which one disagrees. Asking it to
  normalise where the checkpoint ships no statistics raises, rather than quietly doing nothing.

  The 16,384 feature descriptions come down in one request, with
  `protein esm features prepare`. A feature number becomes a name you can read.

- **`protein.xref`, which says what gene a UniProt accession names.** Swiss-Prot fills in a
  taxon id for every entry it hands back, and nothing read it. `gene_stems_for` takes a list
  of accessions and one taxon id, and answers with the gene ids `liulab-genome` maps them to.
  `species_for` turns a taxon id into a species name, or `None` when no set covers it, so a
  caller can check first rather than catch an error.

  **Three species are covered — human, mouse and worm — and Swiss-Prot holds every species.**
  Most entries therefore reach no set at all, and that is kept apart from a real miss. A taxon
  with no set raises `TaxonNotCoveredError`, because the question cannot be asked of it. An
  accession that was asked and matched nothing rides back in the answer's `unresolved` list,
  so a list of accessions never comes back shorter without saying so.

  **The module owns nothing.** `liulab-genome` fetches the data, stores it and records which
  release answered, so there is no new set to prepare, no cache and no new command. One
  direction lives here. Going from a gene to its proteins starts from a species rather than
  from a protein, so it stays a plain `genome.xref` call.

- **A nucleic alphabet, so DNA and RNA get in.** `protein.seq.to_nucleotide_sequence` takes
  the four bases, the eleven IUPAC ambiguity codes and `U`, in either case, and hands back
  biotite's `NucleotideSequence`. `U` becomes `T` and says so; anything else raises the same
  error a bad protein sequence does. There is no `DNA` class and no `RNA` class.

  `Chain.sequence` now answers for a nucleic chain instead of refusing, and refuses only a
  chain that is neither — a ligand or the solvent. `Chain.kind` is unchanged and still says
  which type will come back. The guard that kept DNA away from ESM-C moved to `ESMC.embed`.

- **A structure can say what it was produced from.** `Structure` takes an optional
  `accessions` map, one entry per chain, and `Chain.uniprot` answers from it rather than
  asking SIFTS. A structure folded from a known accession used to answer `()`, which is what
  a deposited entry SIFTS maps nothing to also answers.

  **SIFTS is still the only join.** An accession here is an input the file was written from,
  never a cross-reference read back out of a file — nothing reads `_struct_ref`. Provenance
  does not survive being written, so a prediction reopened from disk answers `()` again.

- **The `esm` environment is now proved to still resolve**, weekly and on demand. Nothing on
  the pull-request path installs it, and what breaks there is the solve rather than the model
  — a PyPI-only `esm`, torch from conda-forge, the CUDA headers Triton's build needs — so no
  GPU is needed to catch it.

- **`pixi.lock` and the action tags have a refresh path.** `liulab-genome` is a git dependency
  named nowhere but the lock, so the next re-solve would take whatever that branch was that
  day. A weekly job re-solves and opens a pull request instead; dependabot covers the actions.

### Changed

- **Four moves that widen nothing.** No import a caller writes changes, no command path moves,
  and no `--json` payload differs. What changes is where things live.

  `protein.search.target` holds the four names that say where a search points, how it is tuned
  and what the query is called. They lived in the MMseqs2 module because that is where they
  were first needed, which left `Structure.search` naming MMseqs2 to describe a Foldseek
  search.

  `protein.msa` is a package of four modules — the `MSA` class, the MMseqs2 recipe, the MUSCLE
  recipe, and the commands — matching `protein.search`. The suite already crossed those four
  seams; only the module had none. `import protein` no longer pulls in a command-line
  framework or biotite's MUSCLE layer.

  `MmseqsLikeTool` goes from ten verbs to seven. `createindex`, `convertalis` and `cluster`
  had no caller and were held alive by their own tests.

  `protein.prepared` holds the half both prepared sets repeated: the status read off the
  completion marker, the cached table read, and the prepare and status commands. SIFTS and the
  SAE feature descriptions each declare only what differs — a source, a field list, a reader.
  `SiftsStatus` and `SaeFeaturesStatus` are one `PreparedStatus`.

### Fixed

- **`main` was red, and the site published from it anyway.** ruff formats Python inside
  Markdown, and `docs/research/` is the one directory whose contract is verbatim quotation;
  four other gates already exempt it and the formatter now does too. The site was also
  deployed from a commit the gate had rejected, because both workflows listened on the same
  push and neither can order itself after the other. The site now waits for the gate, and
  publishes only when it concluded green.

- **CI built a wheel that could not import.** `liulab-genome` is a git dependency, which PyPI
  metadata cannot carry, so it is declared to pixi alone and the wheel raised on its first
  import. The job was green and proved only that hatchling ran. Nothing here is published, so
  the job is gone rather than propped up.

- **The guard on biotite's converters did not name the alignment ones.** A module reading an
  alignment through biotite would have passed every check while turning each selenocysteine
  into cysteine, with no warning and no error. Nothing did that yet, which is why it was a hole
  rather than a live fault, and why it is closed before the alignment code rather than after.

- **The `esm` environment now carries the CUDA headers, so the fused GPU kernels build.**
  They are compiled the first time one runs, against a header the environment did not have,
  and the failure hid its own error — so the fast path looked unavailable and everything ran
  an order of magnitude slower. The environment declares the headers and points the compiler
  at them, by a path inside the environment rather than one belonging to a machine.
