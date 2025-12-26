import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

const MIME_TYPE_PREFERENCE = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4",
];

function pickSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;

  for (const mimeType of MIME_TYPE_PREFERENCE) {
    if (MediaRecorder.isTypeSupported(mimeType)) return mimeType;
  }

  return undefined;
}

function getFileExtension(mimeType?: string): "webm" | "mp4" {
  if (mimeType?.includes("mp4")) return "mp4";
  return "webm";
}

function formatTimestampForFilename(date: Date): string {
  const pad2 = (value: number) => value.toString().padStart(2, "0");
  const yyyy = date.getFullYear();
  const mm = pad2(date.getMonth() + 1);
  const dd = pad2(date.getDate());
  const hh = pad2(date.getHours());
  const min = pad2(date.getMinutes());
  const ss = pad2(date.getSeconds());
  return `${yyyy}-${mm}-${dd}-${hh}${min}${ss}`;
}

type StreamRecorderOptions = {
  filenameBase?: string | null;
  onRecordingSaved?: (info: {
    filenameBase: string;
    filename: string;
    extension: "webm" | "mp4";
    mimeType?: string;
  }) => void;
};

export function useStreamRecorder(
  stream: MediaStream | null,
  options: StreamRecorderOptions = {}
) {
  const preferredFilenameBase = options.filenameBase ?? null;
  const onRecordingSaved = options.onRecordingSaved;
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const recordingStartTimeRef = useRef<number | null>(null);
  const durationTimerRef = useRef<number | null>(null);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const recordingFilenameBaseRef = useRef<string | null>(null);

  const canRecord = useMemo(() => {
    if (!stream) return false;
    if (typeof MediaRecorder === "undefined") return false;
    return stream.getTracks().length > 0;
  }, [stream]);

  const clearDurationTimer = useCallback(() => {
    if (durationTimerRef.current) {
      window.clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
  }, []);

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;

    clearDurationTimer();
    setIsRecording(false);

    if (recorder.state === "inactive") {
      mediaRecorderRef.current = null;
      recordingStreamRef.current = null;
      recordingStartTimeRef.current = null;
      recordingFilenameBaseRef.current = null;
      setRecordingDuration(0);
      return;
    }

    try {
      recorder.stop();
    } catch (error) {
      console.warn("Failed to stop MediaRecorder:", error);
    }
  }, [clearDurationTimer]);

  const startRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") return;

    if (!canRecord || !stream) {
      toast.error("Recording unavailable", {
        description: !stream
          ? "Start a stream to record."
          : "This browser doesn't support recording this stream.",
      });
      return;
    }

    const preferredMimeType = pickSupportedMimeType();
    const options: MediaRecorderOptions = {};
    if (preferredMimeType) {
      options.mimeType = preferredMimeType;
    }

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, options);
    } catch (error) {
      console.error("Failed to create MediaRecorder:", error);
      toast.error("Failed to start recording", {
        description: "Your browser couldn't create a recorder for this stream.",
      });
      return;
    }

    recordedChunksRef.current = [];
    recordingStreamRef.current = stream;
    recordingStartTimeRef.current = performance.now();
    setRecordingDuration(0);

    // Choose and lock a filename base for the duration of this recording.
    const preferredBase = preferredFilenameBase?.trim();
    recordingFilenameBaseRef.current =
      preferredBase || `recording-${formatTimestampForFilename(new Date())}`;

    recorder.ondataavailable = event => {
      if (event.data && event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };

    recorder.onerror = event => {
      console.error("MediaRecorder error:", event);
      toast.error("Recording error", {
        description: "An error occurred while recording.",
      });
      stopRecording();
    };

    recorder.onstop = () => {
      const chunks = recordedChunksRef.current;
      recordedChunksRef.current = [];

      const mimeType = recorder.mimeType || preferredMimeType;
      const extension = getFileExtension(mimeType);
      const filenameBase =
        recordingFilenameBaseRef.current ||
        preferredFilenameBase?.trim() ||
        `recording-${formatTimestampForFilename(new Date())}`;
      const filename = `${filenameBase}.${extension}`;

      if (!chunks.length) {
        toast.error("No recording data", {
          description: "Nothing was captured from the stream.",
        });
      } else {
        const blob = new Blob(chunks, { type: mimeType || "video/webm" });
        const url = URL.createObjectURL(blob);

        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        toast.success("Recording saved", { description: filename });
      }

      onRecordingSaved?.({
        filenameBase,
        filename,
        extension,
        mimeType: mimeType || undefined,
      });

      mediaRecorderRef.current = null;
      recordingStreamRef.current = null;
      recordingStartTimeRef.current = null;
      recordingFilenameBaseRef.current = null;
      clearDurationTimer();
      setRecordingDuration(0);
      setIsRecording(false);
    };

    try {
      recorder.start(1000);
    } catch (error) {
      console.error("Failed to start MediaRecorder:", error);
      toast.error("Failed to start recording", {
        description: "Your browser refused to start recording.",
      });
      recordingFilenameBaseRef.current = null;
      return;
    }

    mediaRecorderRef.current = recorder;
    setIsRecording(true);

    durationTimerRef.current = window.setInterval(() => {
      const startTime = recordingStartTimeRef.current;
      if (!startTime) return;
      const elapsedSeconds = Math.floor((performance.now() - startTime) / 1000);
      setRecordingDuration(elapsedSeconds);
    }, 250);
  }, [
    canRecord,
    clearDurationTimer,
    onRecordingSaved,
    preferredFilenameBase,
    stopRecording,
    stream,
  ]);

  // Stop recording if the stream changes or disappears.
  useEffect(() => {
    if (!isRecording) return;

    const activeStream = recordingStreamRef.current;
    if (!activeStream) return;

    if (!stream || stream !== activeStream) {
      stopRecording();
    }
  }, [isRecording, stopRecording, stream]);

  // Stop recording if any track ends.
  useEffect(() => {
    if (!isRecording) return;

    const activeStream = recordingStreamRef.current;
    if (!activeStream) return;

    const handleEnded = () => stopRecording();
    activeStream.getTracks().forEach(track => {
      track.addEventListener("ended", handleEnded);
    });

    return () => {
      activeStream.getTracks().forEach(track => {
        track.removeEventListener("ended", handleEnded);
      });
    };
  }, [isRecording, stopRecording]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      clearDurationTimer();
      if (mediaRecorderRef.current?.state === "recording") {
        try {
          mediaRecorderRef.current.stop();
        } catch (error) {
          console.warn("Failed to stop MediaRecorder on unmount:", error);
        }
      }
    };
  }, [clearDurationTimer]);

  return {
    canRecord,
    isRecording,
    recordingDuration,
    startRecording,
    stopRecording,
  };
}
