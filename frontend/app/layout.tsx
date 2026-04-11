import type { Metadata } from 'next'
import { Crimson_Pro, Source_Serif_4 } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const crimsonPro = Crimson_Pro({ 
  subsets: ["latin"],
  variable: '--font-serif-heading',
  weight: ['400', '500', '600', '700'],
});

const sourceSerif = Source_Serif_4({ 
  subsets: ["latin"],
  variable: '--font-serif-body',
  weight: ['400', '500', '600'],
});

export const metadata: Metadata = {
  title: 'Bach Propagation Workbench',
  description: 'Generative Counterpoint Settings',
  generator: 'v0.app',
  icons: {
    icon: '/logo.png', // This replaces the complex array
    apple: '/logo.png', // Using the same logo for Apple touch icon
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${crimsonPro.variable} ${sourceSerif.variable} font-serif antialiased`}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
