import { RunDetail } from './run-detail';

export default function RunDetailPage({ params }: { params: { runId: string } }) {
  return <RunDetail runId={params.runId} />;
}
