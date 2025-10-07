import React from 'react';

interface ImpersonationBannerProps {
  clientName: string;
  onExit: () => void;
}

export const ImpersonationBanner: React.FC<ImpersonationBannerProps> = ({ clientName, onExit }) => {
  return (
    <div role="status" className="admin-impersonation-banner">
      <span>Viewing as {clientName}</span>
      <button type="button" onClick={onExit} aria-label="Exit impersonation">
        Exit
      </button>
    </div>
  );
};
