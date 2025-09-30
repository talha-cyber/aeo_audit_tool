import type { Meta, StoryObj } from '@storybook/react';
import { Card, CardContent, CardHeader } from './card';

const meta: Meta<typeof Card> = {
  title: 'UI/Card',
  component: Card,
  render: (args) => (
    <Card {...args}>
      <CardHeader title="Card title" description="Supportive copy in grayscale." />
      <CardContent>
        <p className="text-sm text-muted">Use cards to group related insights and controls.</p>
      </CardContent>
    </Card>
  )
};

export default meta;

type Story = StoryObj<typeof Card>;

export const Default: Story = {
  args: {
    corner: true
  }
};
