import React from 'react';
import clsx from 'clsx'; // For conditional class names; ensure it's installed (npm install clsx)

interface DownloadGifButtonProps {
  gifUrl: string | null | undefined;
  filename?: string;
  buttonText?: string;
  className?: string;
  disabled?: boolean; // To explicitly disable the button, overrides gifUrl check for disabled state
}

const DownloadGifButton: React.FC<DownloadGifButtonProps> = ({
  gifUrl,
  filename = 'qnft_animation.gif', // Default filename
  buttonText = 'Download GIF',
  className,
  disabled = false, // External disabled state
}) => {
  const isEffectivelyDisabled = disabled || !gifUrl;

  // Optional: onClick handler for logging or future fetch/blob implementation
  const handleDownloadClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (isEffectivelyDisabled) {
      event.preventDefault(); // Prevent any action if disabled
      return;
    }
    // For now, this handler is mostly for demonstration or future expansion.
    // The default behavior of the anchor tag with `href` and `download` handles the download.
    console.info(`Download initiated for URL: ${gifUrl} as Filename: ${filename}`);

    // If implementing fetch/blob method in the future, it would go here:
    // event.preventDefault(); // Prevent default anchor behavior
    // async function downloadWithFetch() {
    //   try {
    //     const response = await fetch(gifUrl!);
    //     if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    //     const blob = await response.blob();
    //     const objectUrl = URL.createObjectURL(blob);
    //     const anchor = document.createElement('a');
    //     anchor.href = objectUrl;
    //     anchor.download = filename;
    //     document.body.appendChild(anchor);
    //     anchor.click();
    //     document.body.removeChild(anchor);
    //     URL.revokeObjectURL(objectUrl);
    //   } catch (e) {
    //     console.error('Download failed:', e);
    //     // Handle error (e.g., show a notification to the user)
    //   }
    // }
    // downloadWithFetch();
  };

  return (
    <a
      href={isEffectivelyDisabled ? undefined : gifUrl!} // Set href only if not disabled
      download={!isEffectivelyDisabled ? filename : undefined} // Set download attr only if not disabled
      onClick={handleDownloadClick}
      className={clsx(
        "inline-block px-4 py-2 font-semibold rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-75 transition-colors duration-150 text-center",
        isEffectivelyDisabled
          ? "bg-gray-600 text-gray-400 cursor-not-allowed opacity-70" 
          : "bg-green-600 hover:bg-green-700 text-white cursor-pointer",
        className // Allows additional classes to be passed
      )}
      aria-disabled={isEffectivelyDisabled}
      role="button" // Semantic role
      // If the link is technically active but should behave as disabled, prevent default
      // This is mostly handled by not setting href/download when disabled.
      {...(isEffectivelyDisabled && { onClick: (e) => e.preventDefault() })}
    >
      {buttonText}
    </a>
  );
};

export default DownloadGifButton;
