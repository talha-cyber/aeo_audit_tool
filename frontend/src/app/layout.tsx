import type { Metadata } from 'next';
import './globals.css';
import { Providers } from './providers';

export const metadata: Metadata = {
  title: 'AEO Intelligence Dashboard',
  description: 'Graybox dashboard for the AEO Competitive Intelligence Tool.'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-background text-text antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
