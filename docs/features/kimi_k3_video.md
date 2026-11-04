# Kimi-K3 video input

The vLLM Kimi-K3 model accepts images only. The plugin adds the `video` modality to the Kimi-K3 model class
(`KimiK3ForConditionalGeneration`) when the Kimi-K3 patches are applied. Other models, including Kimi-K2.5 and
Kimi-K2.6, are not changed. No environment variable controls it.

## Request format

Use `video_url` content parts. Plain URLs and base64 `data:video/mp4;base64,...` URLs are decoded by the vLLM video
loader. Unless `--media-io-kwargs` or the request's `media_io_kwargs` say otherwise, the loader keeps at most 8 frames
per second and at most 256 frames.

## Processing

Kimi-K3 publishes no video prompt format; the plugin follows the Kimi-K2.5/K2.6 one:

- the decoded frames are sampled at `min(sample_fps, source fps)` (`sample_fps` comes from the checkpoint, 8.0);
- the frames are cut into chunks of `temporal_merge_kernel_size` (4); the vision tower mean-pools each chunk over time;
- every chunk becomes `hh:mm:ss.fff<|media_begin|>video<|media_content|><|media_pad|>*N<|media_end|>`, where the
  timestamp is the position of the chunk's first frame in the source video.

For example, a 20-frame 864x480 video at 2 fps gives 5 chunks of 558 vision tokens, 2790 in total.

## Warmup

The vision tower is compiled on HPU: in torch.compile mode regional compilation compiles every `nn.ModuleList`,
including `vision_tower.encoder.blocks`, and in lazy mode the tower runs in an HPU graph. A video chunk (for example
T=4 frames, 36x62 patches for 864x480) is a new input shape and creates new graphs. Multimodal warmup precompiles them
from the video shape of `--limit-mm-per-prompt`; a video shape that is not warmed up compiles on the first request. To
warm up the video shape pass it explicitly:

```bash
--limit-mm-per-prompt '{"image": 20, "video": {"count": 1, "width": 864, "height": 480, "num_frames": 20}}'
```

Without a per-modality override the dummy video used for profiling and warmup is 20 frames of 864x480 at 2 fps.

## Failure handling

If the patch cannot be installed, a warning from the `vllm.gaudi.kimi_patches` logger names the patch (`k3_video`)
and the exception; Kimi-K3 then keeps serving images only.
