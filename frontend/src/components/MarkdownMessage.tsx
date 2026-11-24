import React from "react";
import { Check, Copy } from "lucide-react";

import { splitThinkingContent } from "../format";

type Block =
  | { kind: "code"; lang: string; content: string }
  | { kind: "text"; content: string };

function splitCodeBlocks(markdown: string): Block[] {
  const blocks: Block[] = [];
  const fenceRe = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = fenceRe.exec(markdown)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ kind: "text", content: markdown.slice(lastIndex, match.index) });
    }
    blocks.push({ kind: "code", lang: match[1] || "code", content: match[2].replace(/\n$/, "") });
    lastIndex = fenceRe.lastIndex;
  }

  if (lastIndex < markdown.length) {
    blocks.push({ kind: "text", content: markdown.slice(lastIndex) });
  }
  return blocks;
}

function renderInline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const tokenRe = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = tokenRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      nodes.push(<code key={key++}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    lastIndex = tokenRe.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function CodeBlock({ lang, content }: { lang: string; content: string }) {
  const [copied, setCopied] = React.useState(false);

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API may be unavailable in non-secure contexts.
    }
  }

  return (
    <div className="md-code-block">
      <div className="md-code-header">
        <span>{lang}</span>
        <button type="button" className="copy-button" onClick={copyCode} title="Copy code">
          {copied ? <Check size={14} /> : <Copy size={14} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre>
        <code>{content}</code>
      </pre>
    </div>
  );
}

function MarkdownText({ content }: { content: string }) {
  const blocks = splitCodeBlocks(content);
  return (
    <div className="md-body">
      {blocks.map((block, blockIdx) => {
        if (block.kind === "code") {
          return <CodeBlock key={blockIdx} lang={block.lang} content={block.content} />;
        }

        const paragraphs = block.content.split(/\n{2,}/).filter((chunk) => chunk.trim().length > 0);
        return (
          <React.Fragment key={blockIdx}>
            {paragraphs.map((paragraph, paraIdx) => {
              const lines = paragraph.split("\n");
              if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
                return (
                  <ul key={paraIdx}>
                    {lines.map((line, liIdx) => (
                      <li key={liIdx}>{renderInline(line.replace(/^\s*[-*]\s+/, ""))}</li>
                    ))}
                  </ul>
                );
              }
              if (lines.every((line) => /^\s*\d+\.\s+/.test(line))) {
                return (
                  <ol key={paraIdx}>
                    {lines.map((line, liIdx) => (
                      <li key={liIdx}>{renderInline(line.replace(/^\s*\d+\.\s+/, ""))}</li>
                    ))}
                  </ol>
                );
              }
              const heading = /^#{1,3}\s+(.+)$/.exec(paragraph.trim());
              if (heading && lines.length === 1) {
                return <h4 key={paraIdx}>{renderInline(heading[1])}</h4>;
              }
              return (
                <p key={paraIdx}>
                  {lines.map((line, lineIdx) => (
                    <React.Fragment key={lineIdx}>
                      {lineIdx > 0 && <br />}
                      {renderInline(line)}
                    </React.Fragment>
                  ))}
                </p>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function MarkdownMessage({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  const { thinking, answer } = splitThinkingContent(content);
  return (
    <div className="assistant-body">
      {thinking !== null && (
        <details className="thinking-block" open={streaming && !answer}>
          <summary>{streaming && !answer ? "Thinking..." : "Thinking"}</summary>
          <pre className="thinking-trace">{thinking}</pre>
        </details>
      )}
      {answer ? (
        <MarkdownText content={answer} />
      ) : thinking !== null && streaming ? (
        <p className="pending-copy">Drafting answer...</p>
      ) : null}
    </div>
  );
}
