'use client';

import clsx from 'clsx';
import { useEffect, useState } from 'react';

interface Tab<T extends string> {
  id: T;
  label: string;
  content: React.ReactNode;
}

interface TabsProps<T extends string> {
  tabs: Tab<T>[];
  initialTab?: T;
  activeTab?: T;
  onTabChange?: (tab: T) => void;
}

export function Tabs<T extends string>({ tabs, initialTab, activeTab, onTabChange }: TabsProps<T>) {
  const [internalTab, setInternalTab] = useState<T>(initialTab ?? tabs[0].id);
  const currentTab = activeTab ?? internalTab;

  useEffect(() => {
    if (initialTab && !activeTab) {
      setInternalTab(initialTab);
    }
  }, [initialTab, activeTab]);

  const handleTabClick = (tabId: T) => {
    if (onTabChange) {
      onTabChange(tabId);
    } else {
      setInternalTab(tabId);
    }
  };

  return (
    <div>
      <div className="flex items-center gap-2 border-b border-border pb-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => handleTabClick(tab.id)}
            className={clsx(
              'rounded-md px-3 py-2 text-sm font-medium transition-colors',
              currentTab === tab.id
                ? 'bg-elevated text-text shadow-inner'
                : 'text-muted hover:text-text'
            )}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="pt-4">
        {tabs.map((tab) => (
          <div key={tab.id} className={clsx({ hidden: currentTab !== tab.id })}>
            {tab.content}
          </div>
        ))}
      </div>
    </div>
  );
}
