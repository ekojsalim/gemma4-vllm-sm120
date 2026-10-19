# Saved-output quality spot-check

Reviewed the existing GPU responses against their source prompts, without running
new inference. **No obvious cache-related semantic regression was found in this
sample.** Output variation alone is not a reason to block prefix-cache support.
This is an assistant qualitative review, not a blinded evaluation or a labeled
quality benchmark. Private source/output text remains outside the repository.

## Coverage

The three runs contain 76 saved responses. Nine are translations of the same
private Chinese excerpt: four distinct translations after deduplicating repeated
outputs. All four were read against the requested source, surrounding reference
context, glossary and formatting instructions. The remaining responses cover
public Indonesian spans, JSON location checks and one-token decisions; all
unique public outputs were read against the six source paragraphs or the changed
location prompt. Repeated identical outputs do not add independent evidence.

## Private translation

All variants preserve the scene's main events, motivation, identification clues,
character relationships, instructions in the dialogue and closing transition.
Names and the main glossary entities remain consistent. None translates the
reference-only preceding/following context into the answer. No obvious truncation,
repeated passage or invented major event was found.

Differences mostly concern phrasing and intensity. Cache-on variants sometimes
handle an anger idiom and a noisy source term more naturally than the control,
which renders them awkwardly/literally. That does not establish a general quality
advantage, but the changed outputs are not uniformly worse.

Caveats found in both modes:

- All flatten an emphatic comparison about how effortlessly an action is done.
  The main meaning survives, but the stronger comparison is lost.
- All omit embedded advertising/watermark material, including numeric strings
  that also appear in the supplied glossary. This may be desirable cleanup, but
  it is not literal preservation and should be judged under the application's
  intended policy. It is not specific to caching.
- Tone, idiom naturalness and some descriptive specificity vary modestly.

There is one formatting difference worth carrying into future fixtures: cache-on
outputs add an ASCII three-dot scene separator before the closing location/POV
transition, whereas the control does not. The source has a transition phrase but
no explicit decorative divider at that point. The instruction's intended handling
of implied versus explicit scene breaks should determine the score; this is not
by itself evidence of incorrect cache reuse.

## Public spans and verification

All six paragraphs retain their principal facts: destination and empty basket;
bicycle ownership/color and protected letter; route and planned departure;
orchard/bridge and rain damage; gardening and invitation; old photographs and
family connection. Assignments and span IDs match, and no paragraph is replaced
with a sibling's content. Variations such as “mango grove” versus “mango orchard”
and “keep it dry” versus “keep it from getting wet” are acceptable paraphrases.
The indefinite rendering of the grandmother in paragraph five is mildly awkward
but shared by both modes, not a cache-on regression.

JSON responses give the correct location, differing only in whitespace. One-token
decisions give A for the original chapter and B for the modified destination,
then A again when the original is restored (the two final variants include this
changed-input check).

## Implication

Keep the narrower acceptance criterion: preserve task quality and isolation while
saving shared-prefix prefill work. Do not require identical generated strings,
identical speculative acceptance, or a total-latency win on the old translation
fixture. The saved responses reveal no obvious quality blocker for continuing
with representative translation-spans fixtures. They cannot rule out less common
errors, long-context issues or a systematic regression across a larger corpus.
