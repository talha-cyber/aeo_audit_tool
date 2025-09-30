import { ReportReader } from './report-reader';

export default function ReportReaderPage({ params }: { params: { reportId: string } }) {
  return <ReportReader reportId={params.reportId} />;
}
