Here is the Markdown transcription of the PDF content, annotated with page numbers.

# torchao release compatibility table #2919

**Status:** Open
**Author:** vkuzo
**Date:** Sep 2, 2025 (Edited)
**Source:** [Page 1]

This issue describes the compatibility matrix between torchao releases and its dependencies. If you are seeing an error when importing `torchao` that looks like this:

```text
(pytorch_nightly) [vasiliy@devgpu007.eag6 ~/local]$ python -c "import torchao"
Fatal Python error: Aborted
```

then most likely you can resolve this error by ensuring that the `torch` version in your environment is compatible with the `torch` version used to build your `torchao` version.

### Compatibility Matrix

| torchao version | torch version | torch version, torchao's Python API only |
| :--- | :--- | :--- |
| 0.15.0dev (nightly) | 2.10.0dev (nightly) | 2.10.0, 2.9.0, 2.8.0 |
| 0.14.1 | 2.9.0 | 2.9.0, 2.8.0, 2.7.1 |
| 0.13.0 | 2.8.0 | 2.8.0, 2.7.1, 2.6.0 |
| 0.12.0 | 2.7.1, 2.6.0, 2.5.0 | n/a |

---
**[Page 2]**

### fbgemm_gpu

`torchao` has an optional runtime dependency on `fbgemm_gpu`. Please see [FBGEMM Releases](https://docs.pytorch.org/FBGEMM/general/Releases.html) for the compatibility matrix for `fbgemm_gpu`.

**Note:** While `torchao`'s Python API supports multiple `torch` versions, each `fbgemm_gpu` version only supports a single `torch` version. Therefore, if you are using `torchao` together with `fbgemm_gpu`, you should use the `torch` version corresponding to your `fbgemm_gpu` version.

---

## Comments

**jerryzh168** on Sep 10 **[Page 2]**
> any ideas when we can fix the compatibility of torchao nightly with torch? currently it's blocking tests in vllm: [GitHub Link]

**yushangdi** on Sep 10 (Edited) **[Page 2]**
> I'm also having the same issue here. Blocking testing executorch with torch nightly.
> I have torch on latest master (with some python-only local changes) `'2.9.0a0+git3564a8a'`, and torchao `torchao-0.14.0.dev20250909+cu126`.
>
> I have it resolved by installing `pip install fbgemm-gpu-nightly` to override `fbgemm-gpu`. So far importing torchao doesn't error anymore. Not sure if there're any other issues if I actually run anything.

**jerryzh168** on Sep 10 **[Page 3]**
> executorch doesn't need fbgemm-gpu I think? if you use torchao nightly, can you also use torch nightly in ET?

**vkuzo** on Sep 15 **[Page 3]**
> we also need to update the following check [GitHub Link to \_\_init\_\_.py] for PyTorch 2.10.x, as `str(torch.__version__) >= "2.9"` will not work properly for PyTorch 2.10

**vkuzo** on Sep 19 **[Page 3]**
> @liangel-02 , [GitHub Link to FBGEMM PR #1900] might be relevant here - this is fbgemm fixing the same issue in their repo

**NeonSludge** on Oct 15 **[Page 4]**
> This makes it very hard to use torchao with latest versions of Ray Serve/vLLM:
> ```
> Skipping import of cpp extensions due to incompatible torch version 2.8.0+cu128 for torchao version 0.14.0
> ```

**vkuzo** on Oct 15 **[Page 4]**
> Hi folks, thank you for reporting, we are looking into this and will provide an update soon. Note that unless you actually need c++ or CUDA kernels that ship with torchao, you can ignore the warning and use the Python-only APIs without issues.

**atalman** on Oct 15 **[Page 4]**
> PyTorch release 2.9.0 eta date is today : [Dev Discuss Link]

**danielvegamyhre** (3 weeks ago) **[Page 5]**
> Update - We are pausing the release plans until the PyTorch CI is turned back on. We have it disabled temporarily as a mitigation for a widespread GitHub security issue: pytorch/pytorch#169033
> We're hoping this will be resolved in the coming days, and we'll post updates here periodically.

**steveepreston** (3 weeks ago) **[Page 5]**
> damn, if possible please silent it by default

**danielvegamyhre** (3 weeks ago) **[Page 5]**
> Update: we are proceeding with the release thursday/friday

**FurkanGozukara** (2 weeks ago) **[Page 5]**
> it is amazing it shows to check this thread

**huunghiagv06-droid** (2 weeks ago) **[Page 6]**
> goods

**NSR-007** (last week) **[Page 6]**
> any solution yet?
> even torchao 0.14.1+cu128 gives the error...:
> "Skipping import of cpp extensions due to incompatible torch version 2.8.0+cu128 for torchao version 0.14.1+cu128"
> does it affect peformance in ComfyUI?

**steveepreston** (last week) **[Page 6]**
> **does it affect peformance?**
> This is main question!
> nothing is clear
> we get this warning even when not directly importing `torch ao`, mean some other package imported it, but the `torch ao` throws this

**NSR-007** (last week) **[Page 6]**
> Fix seems to be `torchao==0.12.0+cu128` with `torch==2.7.1`, matches with `xformers==0.0.31.post1`
>
> ```bash
> pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
> pip install torchao==0.12.0+cu128 --index-url https://download.pytorch.org/whl/cu128
> pip install xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128
> ```
>
> **[Page 7]**
> Only slight speed increase though, comparing two 3090ti running each a ComfyUI instance running Wan2.2, however the other ComfyUI is on torch==2.8 so has Skipping cpp ext...):
>
> *   ComfyUI + torch2.7.1 = Prompt executed in 725.08s / 00:12:05 (no Error "Skipping cpp extenson...")(no Nvidia Apex)
> *   ComfyUI + torch2.8.0 = Prompt executed in 699.32s / 00:11:39 (Nvidia Apex installed)
>
> (3090ti Power limit = 300 watt, max. 1875 Mhz)
> BTW with torch 2.9 I have to lower the resolutions to run Wan2.2...

---
**Timeline & Metadata**
*   **Sep 3:** `vkuzo` mentioned issue #2901 (Aborted (core dumped) when importing v0.13.0 RC) **[Page 2]**
*   **Sep 19:** `vkuzo` assigned `liangel-02` **[Page 3]**
*   **Sep 22:** `liangel-02` mentioned in PR #3042 (generalize torch compatibility check) **[Page 3]**
*   **Oct 7:** `liangel-02` mentioned in PR #3130 (Update version to 0.15.0) **[Page 3]**
*   **Last week:** `jerryzh168` mentioned PR #3516 (Making torchao ABI compatible and moving closer to python only) **[Page 7]**
