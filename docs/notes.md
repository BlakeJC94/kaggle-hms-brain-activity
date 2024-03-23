# Notes

Terminology from provided reference:

> Generalized (G): any bilaterally synchronous and symmetric pattern even if it has a restricted
> field (e.g., bifrontal).

> Lateralized (L): unilateral (Fig. 14); OR bilateral but clearly and consistently higher amplitude
> in one hemisphere (bilateral asymmetric; OR bilateral but with a consistent lead-in from the same
> side (bilateral asynchronous) (Fig. 16). This includes focal, regional, and hemispheric patterns.

> Periodic Discharges (PDs):
>
> * Periodic: Repetition of a waveform with relatively uniform morphology and duration with a
>   clearly discernible inter-discharge interval between consecutive waveforms and recurrence of the
>   waveform at nearly regular intervals. “Nearly regular intervals” is defined as having a cycle
>   length (i.e., period) varying by ,50% from one cycle to the next in most (.50%) cycle pairs.
>
> * Discharges: Waveforms lasting ,0.5 seconds, regardless of number of phases, or waveforms >= 0.5
>   seconds with no more than 3 phases. This is as opposed to Bursts, defined as waveforms lasting
>   <= 0.5 seconds and having at least 4 phases. Discharges and bursts must clearly stand out from
>   the background activity.

> Rhythmic Delta Activity (RDA):
> * Rhythmic: Repetition of a waveform with relatively uniform morphology and duration and without
>   an interval between consecutive waveforms (Fig. 21). The duration of one cycle (i.e., the
>   period) of the rhythmic pattern should vary by ,50% from the duration of the subsequent cycle
>   for most (.50%) cycle pairs to qualify as rhythmic. An example of a rhythmic pattern would be a
>   sinusoidal waveform, although there are other examples; a pattern can be sharp at the top and/or
>   the bottom of the waveform and still be rhythmic (but would no longer be sinusoidal). Irregular
>   or polymorphic delta should not be reported as RDA.
> * RDA: Rhythmic activity 0.5 to <=4.0 Hz.


Getting things running on old version of cuda
```bash
$ rye run pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
```
