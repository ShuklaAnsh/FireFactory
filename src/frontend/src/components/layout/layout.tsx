import React from 'react';

export const Layout = (props: { children?: React.ReactNode }) => {
  return (
    <>
      <div>
        <main>
          {props.children}
        </main>
      </div>
    </>
  );
}