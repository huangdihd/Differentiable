import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';

/**
 * Renders trusted course copy as Markdown with GFM and KaTeX math.
 * Raw HTML stays disabled so translations cannot inject DOM nodes or scripts.
 */
export function MathText({ text }: { text: string }) {
  return (
    <div className="markdown-copy">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
        rehypePlugins={[[rehypeKatex, { strict: 'ignore', throwOnError: false }]]}
        skipHtml
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
