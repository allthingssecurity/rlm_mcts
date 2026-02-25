import React, { useState } from "react";
import { VideoInfo } from "../types";

interface Props {
  videos: VideoInfo[];
  onTranscribed: (videos: VideoInfo[]) => void;
}

export default function VideoInput({ videos, onTranscribed }: Props) {
  const [urls, setUrls] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTranscribe = async () => {
    const urlList = urls
      .split(/[\n,]/)
      .map((u) => u.trim())
      .filter(Boolean);

    if (urlList.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: urlList }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      onTranscribed(data.videos);
    } catch (e: any) {
      setError(e.message || "Failed to transcribe");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Video Input
      </h2>

      <textarea
        className="w-full bg-gray-800 text-gray-200 rounded px-3 py-2 text-sm font-mono
                   placeholder-gray-600 border border-gray-700 focus:border-blue-500
                   focus:outline-none resize-none"
        rows={3}
        placeholder="Paste YouTube URL(s), one per line or comma-separated"
        value={urls}
        onChange={(e) => setUrls(e.target.value)}
        disabled={loading}
      />

      <button
        className="mt-2 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                   disabled:text-gray-500 text-white font-medium py-2 px-4 rounded text-sm
                   transition-colors"
        onClick={handleTranscribe}
        disabled={loading || !urls.trim()}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            Transcribing...
          </span>
        ) : (
          "Transcribe"
        )}
      </button>

      {error && (
        <p className="mt-2 text-red-400 text-xs">{error}</p>
      )}

      {/* Video list */}
      {videos.length > 0 && (
        <div className="mt-3 space-y-2">
          {videos.map((v, i) => (
            <div
              key={v.video_id || i}
              className={`p-2 rounded text-xs ${
                v.error
                  ? "bg-red-900/30 border border-red-800"
                  : "bg-gray-800 border border-gray-700"
              }`}
            >
              {v.error ? (
                <p className="text-red-400">Error: {v.error}</p>
              ) : (
                <>
                  <p className="font-medium text-gray-200 truncate">
                    {v.title}
                  </p>
                  <p className="text-gray-500 mt-1">
                    {v.channel} &middot;{" "}
                    {Math.floor((v.duration || 0) / 60)}m &middot;{" "}
                    {((v.transcript_chars || 0) / 1000).toFixed(0)}K chars
                  </p>
                  {v.transcript_preview && (
                    <p className="text-gray-600 mt-1 line-clamp-2">
                      {v.transcript_preview}
                    </p>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
