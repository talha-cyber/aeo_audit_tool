import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './button';

const meta: Meta<typeof Button> = {
  title: 'UI/Button',
  component: Button,
  args: {
    children: 'Primary action'
  }
};

export default meta;

type Story = StoryObj<typeof Button>;

export const Primary: Story = {};

export const Ghost: Story = {
  args: {
    variant: 'ghost',
    children: 'Ghost action'
  }
};
