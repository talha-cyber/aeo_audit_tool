import React from 'react';

interface AdminGateProps {
  isAdmin: boolean;
  children: React.ReactNode;
}

export const AdminGate: React.FC<AdminGateProps> = ({ isAdmin, children }) => {
  if (!isAdmin) {
    return null;
  }
  return <>{children}</>;
};
